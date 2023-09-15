import copy
import math
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from mmdet.models.task_modules import (AssignResult, PseudoSampler,
                                       build_assigner, build_bbox_coder,
                                       build_sampler)
from mmdet.models.utils import multi_apply
#from mmdet.models.utils import inverse_sigmoid
from mmengine.structures import InstanceData

from mmdet3d.models import circle_nms, draw_heatmap_gaussian, gaussian_radius
from mmdet3d.registry import MODELS
from mmdet3d.models.dense_heads.centerpoint_head import SeparateHead

from mmdet3d.structures import xywhr2xyxyr
from mmdet3d.models.layers import nms_bev
from .utils import draw_bool_heatmap

def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y

@MODELS.register_module()
class MFFusionHead(nn.Module):

    def __init__(self,
        num_proposals=128,
        in_channels=128 * 3,
        hidden_channel=128,
        num_classes=4,
        # config for Transformer
        decoder_layer=dict(),
        num_heads=8,
        nms_kernel_size=1,
        bn_momentum=0.1,
        # config for FFN
        common_heads=dict(),
        num_heatmap_convs=2,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        bias='auto',
        # loss
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean'),
        loss_heatmap=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_frontmap=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        # others
        train_cfg=None,
        test_cfg=None,
        bbox_coder=None,
                 ) -> None:
        super(MFFusionHead, self).__init__()

        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.bn_momentum = bn_momentum
        self.nms_kernel_size = nms_kernel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.pc_range = train_cfg['point_cloud_range']

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_frontmap = MODELS.build(loss_frontmap)
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_heatmap = MODELS.build(loss_heatmap)

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.sampling = False

        self.pos_embed_channel = hidden_channel
        self.focus_ratio = 0.5

        # a shared convolution
        self.shared_conv = build_conv_layer(
            dict(type='Conv2d'),
            in_channels,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        frontpoint_layers = []
        frontpoint_layers.append(
            ConvModule(
                hidden_channel,
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
            ))
        frontpoint_layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                hidden_channel,
                1,
                kernel_size=3,
                padding=1,
                bias=bias,
            ))
        self.frontpoint_head = nn.Sequential(*frontpoint_layers)

        heatmap_layers = []
        heatmap_layers.append(
            ConvModule(
                hidden_channel,
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
            ))
        heatmap_layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                hidden_channel,
                num_classes,
                kernel_size=3,
                padding=1,
                bias=bias,
            ))
        self.heatmap_head = nn.Sequential(*heatmap_layers)
        #self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)

        # transformer decoder layers for object query with LiDAR feature
        self.bev_embedding = nn.Sequential(
            nn.Linear(hidden_channel * 2, hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel)
        )
        #self.decoder = nn.ModuleList()
        self.decoder = MODELS.build(decoder_layer)

        # Prediction Head
        common_heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
        self.prediction_head = SeparateHead( hidden_channel,
                                             common_heads,
                                             conv_cfg=conv_cfg,
                                             norm_cfg=norm_cfg,
                                             bias=bias,
            )
        
        self.init_weights()
        self._init_assigner_sampler()

        self.bev_pos = self.coords_bev()

    def coords_bev(self):
        cfg = self.train_cfg if self.train_cfg else self.test_cfg
        # Position Embedding for Cross-Attention, which is re-used during training # noqa: E501
        x_size = torch.div(cfg['grid_size'][0], cfg['out_size_factor'], rounding_mode='floor')
        y_size = torch.div(cfg['grid_size'][1], cfg['out_size_factor'], rounding_mode='floor')
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)
        coord_base = coord_base.view(2, -1).permute(1, 0) # (H*W, 2)
        return coord_base
    
    def pos2embed(self, pos, num_pos_feats=128, temperature=10000):
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = 2 * (dim_t // 2) / num_pos_feats + 1
        pos_x = pos[..., 0, None] / dim_t
        pos_y = pos[..., 1, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        posemb = torch.cat((pos_y, pos_x), dim=-1)
        return posemb

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        if hasattr(self, 'query'):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum
    
    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        if self.sampling:
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)
        else:
            self.bbox_sampler = PseudoSampler()
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                build_assigner(res) for res in self.train_cfg.assigner
            ]
            
    def prepare_for_dn(self, batch_size, reference_points, img_metas):
        if self.training:
            targets = [torch.cat((img_meta['gt_bboxes_3d']._data.gravity_center, img_meta['gt_bboxes_3d']._data.tensor[:, 3:]),dim=1) for img_meta in img_metas ]
            labels = [img_meta['gt_labels_3d']._data for img_meta in img_metas ]
            known = [(torch.ones_like(t)).cuda() for t in labels]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known)
            known_num = [t.size(0) for t in targets]
            labels = torch.cat([t for t in labels])
            boxes = torch.cat([t for t in targets])
            batch_idx = torch.cat([torch.full((t.size(0), ), i) for i, t in enumerate(targets)])

            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)
            # add noise
            groups = min(self.scalar, self.num_query // max(known_num))
            known_indice = known_indice.repeat(groups, 1).view(-1)
            known_labels = labels.repeat(groups, 1).view(-1).long().to(reference_points.device)
            known_labels_raw = labels.repeat(groups, 1).view(-1).long().to(reference_points.device)
            known_bid = batch_idx.repeat(groups, 1).view(-1)
            known_bboxs = boxes.repeat(groups, 1).to(reference_points.device)
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()
            
            if self.bbox_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.bbox_noise_trans
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
                known_bbox_center += torch.mul(rand_prob,
                                            diff) * self.bbox_noise_scale
                known_bbox_center[..., 0:1] = (known_bbox_center[..., 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
                known_bbox_center[..., 1:2] = (known_bbox_center[..., 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
                known_bbox_center[..., 2:3] = (known_bbox_center[..., 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
                known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
                mask = torch.norm(rand_prob, 2, 1) > self.split
                known_labels[mask] = sum(self.num_classes)

            single_pad = int(max(known_num))
            pad_size = int(single_pad * groups)
            padding_bbox = torch.zeros(pad_size, 3).to(reference_points.device)
            padded_reference_points = torch.cat([padding_bbox, reference_points], dim=0).unsqueeze(0).repeat(batch_size, 1, 1)

            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(groups)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(reference_points.device)

            tgt_size = pad_size + self.num_query
            attn_mask = torch.ones(tgt_size, tgt_size).to(reference_points.device) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(groups):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == groups - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True

            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'known_labels_raw': known_labels_raw,
                'know_idx': know_idx,
                'pad_size': pad_size
            }
            
        else:
            padded_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict

    def _bev_query_embed(self, ref_points):
        bev_embeds = self.bev_embedding(self.pos2embed(ref_points, num_pos_feats=self.pos_embed_channel))
        return bev_embeds

    def query_embed(self, ref_points, img_metas):
        bev_embeds = self._bev_query_embed(ref_points)
        return bev_embeds
    
    @torch.no_grad()  
    def get_front_points( self, points, frontboolmap_flatten ):
        cfg = self.train_cfg if self.train_cfg else self.test_cfg
        x_normal = (points[..., 0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        y_normal = (points[..., 1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        x_size = torch.div(cfg['grid_size'][0], cfg['out_size_factor'], rounding_mode='floor')
        y_size = torch.div(cfg['grid_size'][1], cfg['out_size_factor'], rounding_mode='floor')
        x_idx = ( x_normal*x_size ).long()
        y_idx = ( y_normal*y_size ).long()
        idx = y_idx * x_size + x_idx
        front_points_mask = frontboolmap_flatten.gather( index = idx, dim = -1 )
        #print( front_points_mask.shape )
        return points[front_points_mask, :3]
    
    @torch.no_grad()    
    def get_img_tokens_reference_points( self, front_points, img_feats, meta ):
        N, _ = front_points.shape
        V, C, H, W = img_feats.shape
        #print(meta.keys())
        #pad_h, pad_w, _ = meta['pad_shape'][0]
        #print( pad_h, pad_w )
        reference_points = img_feats.new_zeros( V, H, W, 3 )

        lidars2imgs = torch.from_numpy(meta['lidar2img']).float().to(front_points.device)
        #imgs2lidars = torch.linalg.inv(lidars2imgs)

        proj_points = torch.einsum('nd, vcd -> vnc', torch.cat([front_points, front_points.new_ones(*front_points.shape[:-1], 1)], dim=-1), lidars2imgs)
        #print( proj_points.shape )
        proj_points_clone = proj_points.clone()
        z_mask = (proj_points[..., 2:3].detach() > 0) & ( proj_points[..., 2:3].detach() < 1000.0 )
        proj_points_clone[..., :3] = proj_points[..., :3] / (proj_points[..., 2:3].detach() + z_mask * 1e-6 - (~z_mask) * 1e-6)
        w_idx = torch.div( proj_points_clone[..., 0], 16, rounding_mode='floor').long()
        h_idx = torch.div( proj_points_clone[..., 1], 16, rounding_mode='floor').long()
        mask = ( w_idx < W ) & ( w_idx >= 0) & ( h_idx < H ) & ( h_idx >= 0 )
        mask &= z_mask.squeeze(-1)
        depth = proj_points[..., 2] 
        w_idx[~mask], h_idx[~mask], depth[~mask] = W, H, 1000.0
        #print( w_idx.shape )
        ranks = h_idx*W + w_idx
        sort = ( ranks + depth/1001. ).argsort(dim = -1)
        w_idx = w_idx.gather(index = sort, dim = -1)
        h_idx = h_idx.gather(index = sort, dim = -1)
        depth = depth.gather(index = sort, dim = -1)
        ranks = ranks.gather(index = sort, dim = -1)
        kept = ranks.new_ones(*ranks.shape, dtype=torch.bool)
        kept[:, 1:] = (ranks[:, 1:] != ranks[:, :-1])
        #print( kept.shape )
        
        #print( w_idx.shape )
        #print( h_idx.shape )
        #assert w_idx.shape == (V, W+1)
        #assert h_idx.shape == (V, H+1) 
        for i in range(V):
            w_idx_i, h_idx_i = w_idx[i, kept[i]], h_idx[i, kept[i]]
            #print( w_idx_i.shape )
            #print(w_idx_i.device)
            front_points_i = front_points[kept[i]]
            #print( front_points_i.shape )
            reference_points[ i, h_idx_i[:-1], w_idx_i[:-1]] = front_points_i[:-1]
        
        #reference_points = reference_points.view( V, W, H, 3).permute( 0, 3, 1, 2 )
        return reference_points

    def forward_single(self, points, pts_feats, img_feats, img_metas):

        batch_size, _, H, W = pts_feats.shape
        valid_token_nums = H*W
        feat_pc = self.shared_conv(pts_feats)

        #################################
        # image to BEV
        #################################
        feat_flatten = feat_pc.view(batch_size,
                                    feat_pc.shape[1],
                                    -1)  # [BS, C, H*W]
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(feat_flatten.device)

        key_embed = feat_flatten.transpose( 1, 2 )
        key_pos_embed = self.bev_embedding(self.pos2embed( bev_pos, num_pos_feats = self.pos_embed_channel ))


        #################################
        # choose the front points
        #################################
        focus_token_nums = min( int(valid_token_nums * self.focus_ratio) + 1, valid_token_nums ) 
        with torch.autocast('cuda', enabled=False):
            front_map_score = self.frontpoint_head(feat_pc.float())
        frontmap_flatten = front_map_score.detach().sigmoid().view(batch_size, -1)

        front_idx = frontmap_flatten.argsort(
                    dim=-1, descending=True)[..., :focus_token_nums]
        frontboolmap_flatten= frontmap_flatten.new_zeros(*frontmap_flatten.shape, dtype=torch.bool)

        #print( img_feats.shape )
        for i in range(batch_size):
            frontboolmap_flatten[i, front_idx[i]] = True
            front_points = self.get_front_points( points[i], frontboolmap_flatten[i] )
            img_tokens_reference_points = self.get_img_tokens_reference_points( 
                front_points, img_feats[i], img_metas[i] )
        frontboolmap = frontboolmap_flatten.view(batch_size, H, W)

#
        #front_feat_flatten = feat_flatten.gather( index = front_idx[:, None, :].expand(
        #    -1, feat_flatten.shape[1], -1), dim = -1 )
        #front_pos =  bev_pos.gather(
        #    index= front_idx[..., None].expand(
        #        -1, -1, bev_pos.shape[-1]), dim=1 )
#
        #key_embed = front_feat_flatten.transpose( 1, 2 )
        #key_pos_embed = self.bev_embedding(self.pos2embed( front_pos, num_pos_feats = self.pos_embed_channel ))

        #################################
        # query initialization
        #################################
        with torch.autocast('cuda', enabled=False):
            dense_heatmap = self.heatmap_head(feat_pc.float())
        heatmap = dense_heatmap.detach().sigmoid()
        #heatmap = heatmap * frontboolmap[:, None, ...]

        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        # equals to nms radius = voxel_size * out_size_factor * kenel_size
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
        local_max[:, :, padding:(-padding),
                  padding:(-padding)] = local_max_inner
        # for Pedestrian & Traffic_cone in nuScenes
        if self.train_cfg['dataset'] == 'nuScenes':
            local_max[:, 8, ] = F.max_pool2d(
                heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            local_max[:, 9, ] = F.max_pool2d(
                heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        elif self.train_cfg[
                'dataset'] == 'Waymo':  # for Pedestrian & Cyclist in Waymo
            local_max[:, 1, ] = F.max_pool2d(
                heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            local_max[:, 2, ] = F.max_pool2d(
                heatmap[:, 2], kernel_size=1, stride=1, padding=0)
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

        # top num_proposals among all classes
        top_proposals = heatmap.view(batch_size, -1).argsort(
            dim=-1, descending=True)[..., :self.num_proposals]
        top_proposals_class = torch.div(top_proposals, heatmap.shape[-1], rounding_mode='floor')
        top_proposals_index = top_proposals % heatmap.shape[-1]
        #query_feat = fusion_feat_flatten.gather(
        #    index=top_proposals_index[:, None, :].expand(
        #        -1, fusion_feat_flatten.shape[1], -1),
        #    dim=-1,
        #)
        #self.query_labels = top_proposals_class


        ## add category embedding
        #one_hot = F.one_hot(
        #    top_proposals_class,
        #    num_classes=self.num_classes).permute(0, 2, 1)
        #query_cat_encoding = self.class_encoding(one_hot.float())

        query_pos =  bev_pos.gather(
            index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(
                -1, -1, bev_pos.shape[-1]),
            dim=1,
        )
        query_embed = self.query_embed( query_pos, img_metas )
        
        #query_embed += query_cat_encoding
        #key_pos_embed = self.bev_embedding(self.pos2embed( bev_pos, num_pos_feats = self.pos_embed_channel )
        #                )
        #key_embed = feat_flatten.transpose( 1, 2 )

        #################################
        # transformer decoder layer (Fusion feature as K,V)
        #################################
        query_embed = self.decoder(
            query_embed,
            key= key_embed,
            query_pos=None,
            key_pos=key_pos_embed ) 

        # Prediction
        ret_dict = self.prediction_head(query_embed.transpose(1,2))
        ret_dict['center'] = ret_dict['center'] + query_pos.permute(
            0, 2, 1)   

        ret_dict['query_heatmap_score'] = heatmap.gather(
            index=top_proposals_index[:,
                                      None, :].expand(-1, self.num_classes,
                                                      -1),
            dim=-1,
        )  # [bs, num_classes, num_proposals]
        ret_dict['dense_heatmap'] = dense_heatmap
        ret_dict['frontmap'] = front_map_score
        ret_dict['frontboolmap'] = frontboolmap

        return [ret_dict]
        
    
    def forward(self, points, pts_feats, img_feats, metas):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.
        Returns:
            tuple(list[dict]): Output results. first index by level, second
            index by layer
        """
        #if isinstance(points, torch.Tensor):
        #    points = [points]
        if isinstance(pts_feats, torch.Tensor):
            pts_feats = [pts_feats]
        if isinstance(img_feats, torch.Tensor):
            img_feats = [img_feats]
        res = multi_apply(self.forward_single, [points], pts_feats, img_feats, [metas])
        assert len(res) == 1, 'only support one level features.'
        return res
    
    def get_targets(self, batch_gt_instances_3d: List[InstanceData],
                    preds_dict: dict):
        """Generate training targets.
        Args:
            batch_gt_instances_3d (List[InstanceData]):
            preds_dict (dict): The prediction results. The dict contains
                predictions of one mini-batch:
                - center: (bs, 2, num_proposals)
                - height: (bs, 1, num_proposals)
                - dim: (bs, 3, num_proposals)
                - rot: (bs, 2, num_proposals)
                - vel: (bs, 2, num_proposals)
                - cls_logit: (bs, num_classes, num_proposals)
                - query_score: (bs, num_classes, num_proposals)
                - heatmap: The original heatmap before fed into transformer
                    decoder, with shape (bs, 10, h, w)
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.
                - torch.Tensor: classification target.  [BS, num_proposals]
                - torch.Tensor: classification weights (mask)
                    [BS, num_proposals]
                - torch.Tensor: regression target. [BS, num_proposals, 8]
                - torch.Tensor: regression weights. [BS, num_proposals, 8]
        """
        # change preds_dict into list of dict (index by batch_id)
        # preds_dict['center'].shape [bs, 3, num_proposal]
        list_of_pred_dict = []
        for batch_idx in range(len(batch_gt_instances_3d)):
            pred_dict = {}
            for key in preds_dict.keys():
                pred_dict[key] = preds_dict[key][batch_idx:batch_idx + 1]
            list_of_pred_dict.append(pred_dict)


        assert len(batch_gt_instances_3d) == len(list_of_pred_dict)
        res_tuple = multi_apply(
            self.get_targets_single,
            batch_gt_instances_3d,
            list_of_pred_dict,
        )
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        ious = torch.cat(res_tuple[4], dim=0)
        num_pos = np.sum(res_tuple[5])
        matched_ious = np.mean(res_tuple[6])
        heatmap = torch.cat(res_tuple[7], dim=0)
        frontmap = torch.cat(res_tuple[8], dim=0)
        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            heatmap,
            frontmap,
        )

    def get_targets_single(self, gt_instances_3d, preds_dict):
        """Generate training targets for a single sample.
        Args:
            gt_instances_3d (:obj:`InstanceData`): ground truth of instances.
            preds_dict (dict): dict of prediction result for a single sample.
        Returns:
            tuple[torch.Tensor]: Tuple of target including 
                the following results in order.
                - torch.Tensor: classification target.  [1, num_proposals]
                - torch.Tensor: classification weights (mask) [1,
                    num_proposals] # noqa: E501
                - torch.Tensor: regression target. [1, num_proposals, 8]
                - torch.Tensor: regression weights. [1, num_proposals, 8]
                - torch.Tensor: iou target. [1, num_proposals]
                - int: number of positive proposals
                - torch.Tensor: heatmap targets.
        """
        # 1. Assignment
        gt_bboxes_3d = gt_instances_3d.bboxes_3d
        gt_labels_3d = gt_instances_3d.labels_3d
        num_proposals = preds_dict['center'].shape[-1]

        # get pred boxes, carefully ! don't change the network outputs
        score = copy.deepcopy(preds_dict['heatmap'].detach())
        center = copy.deepcopy(preds_dict['center'].detach())
        height = copy.deepcopy(preds_dict['height'].detach())
        dim = copy.deepcopy(preds_dict['dim'].detach())
        rot = copy.deepcopy(preds_dict['rot'].detach())
        if 'vel' in preds_dict.keys():
            vel = copy.deepcopy(preds_dict['vel'].detach())
        else:
            vel = None

        boxes_dict = self.bbox_coder.decode(
            score, rot, dim, center, height,
            vel)  # decode the prediction to real world metric bbox
        bboxes_tensor = boxes_dict[0]['bboxes']
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)


        assign_result = None
        if self.train_cfg.assigner.type == 'HungarianAssigner3D':
            assign_result = self.bbox_assigner.assign(
                bboxes_tensor,
                gt_bboxes_tensor,
                gt_labels_3d,
                score,
                self.train_cfg,
            )
        elif self.train_cfg.assigner.type == 'HeuristicAssigner':
            assign_result = self.bbox_assigner.assign(
                bboxes_tensor,
                gt_bboxes_tensor,
                None,
                gt_labels_3d,
                #self.query_labels[batch_idx],
            )
        else:
            raise NotImplementedError

        # 1. Sampling. Compatible with the interface of `PseudoSampler` in
        # mmdet.
        gt_instances, pred_instances = InstanceData(
            bboxes=gt_bboxes_tensor), InstanceData(priors=bboxes_tensor)
        sampling_result = self.bbox_sampler.sample(assign_result,
                                                   pred_instances,
                                                   gt_instances)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        assert len(pos_inds) + len(neg_inds) == num_proposals

        # 2. Create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.bbox_coder.code_size
                                    ]).to(center.device)
        bbox_weights = torch.zeros([num_proposals, self.bbox_coder.code_size
                                    ]).to(center.device)
        ious = assign_result.max_overlaps
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(
            num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes

        # both pos and neg have classification loss, only pos has regression
        # and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(
                sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # # compute dense heatmap targets
        device = labels.device
        gt_bboxes_3d = torch.cat(
            [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]],
            dim=1).to(device)
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])
        
        feature_map_size = (torch.div(grid_size[:2], self.train_cfg['out_size_factor'],
                            rounding_mode='floor'))  # [x_len, y_len]


        heatmap = gt_bboxes_3d.new_zeros(self.num_classes, feature_map_size[1],
                                         feature_map_size[0])
        frontmap = gt_bboxes_3d.new_zeros(feature_map_size[1], feature_map_size[0],
                                          dtype=int)
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / voxel_size[0] / self.train_cfg['out_size_factor']
            length = length / voxel_size[1] / self.train_cfg['out_size_factor']
            if width > 0 and length > 0:
                radius_heatmap = gaussian_radius(
                    (length, width),
                    min_overlap=self.train_cfg['gaussian_overlap'])
                radius_heatmap = max(self.train_cfg['min_radius'], int(radius_heatmap))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = ((x - pc_range[0]) / voxel_size[0] /
                          self.train_cfg['out_size_factor'])
                coor_y = ((y - pc_range[1]) / voxel_size[1] /
                          self.train_cfg['out_size_factor'])

                center = torch.tensor([coor_x, coor_y],
                                      dtype=torch.float32,
                                      device=device)
                center_int = center.to(torch.int32)

                # original
                # draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]], center_int, radius) # noqa: E501
                # NOTE: fix
                draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]],
                                      center_int[[1, 0]], radius_heatmap)
                radius_frontmap = int(math.sqrt( length*length + width*width))+1
                draw_bool_heatmap(frontmap, center_int[[1, 0]], radius_frontmap )

        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        return (
            labels[None],
            label_weights[None],
            bbox_targets[None],
            bbox_weights[None],
            ious[None],
            int(pos_inds.shape[0]),
            float(mean_iou),
            heatmap[None],
            frontmap[None],
        )
    
    def loss(self, points, pts_feats, img_feats, batch_data_samples):
        """Loss function for CenterHead.

        Args:
            batch_feats (): Features in a batch.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.
        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        if pts_feats is None:
            pts_feats = [None]
        if img_feats is None:
            img_feats = [None]
        batch_input_metas, batch_gt_instances_3d = [], []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
        preds_dicts = self(points, pts_feats, img_feats, batch_input_metas)
        loss = self.loss_by_feat(preds_dicts, batch_gt_instances_3d)

        return loss

    def loss_by_feat(self, preds_dicts: Tuple[List[dict]],
                     batch_gt_instances_3d: List[InstanceData], *args,
                     **kwargs):
        (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            heatmap,
            frontmap,
        ) = self.get_targets( batch_gt_instances_3d, preds_dicts[0][0] )
        if hasattr(self, 'on_the_image_mask'):
            label_weights = label_weights * self.on_the_image_mask
            bbox_weights = bbox_weights * self.on_the_image_mask[:, :, None]
            num_pos = bbox_weights.max(-1).values.sum()
        loss_dict = dict()
        preds_dict = preds_dicts[0][0]

        # compute frontmap loss
        loss_frontmap = self.loss_frontmap( 
            preds_dict['frontmap'].reshape(-1).float(),
            frontmap.reshape(-1),
            #avg_factor=max(num_pos, 1)
        )
        loss_dict['loss_frontmap'] = loss_frontmap
        loss_dict['frontmap_ratio'] = frontmap.sum().float() / frontmap.numel()

        front_recall_map = (frontmap & ( frontmap == preds_dict['frontboolmap'] )).int()
        frontmap_recall = (front_recall_map.sum(dim=(1,2)).float() / frontmap.sum(dim=(1,2))).mean()
        loss_dict['frontmap_recall'] = frontmap_recall
        #front_recall_map_from_heatmap = (frontmap & ( frontmap == preds_dict['frontboolmap_from_heatmap'] )).int()
        #frontmap_from_heatmap_recall = (front_recall_map_from_heatmap.sum(dim=(1,2)).float() / frontmap.sum(dim=(1,2))).mean()
        #loss_dict['frontmap_from_heatmap_recall'] = frontmap_from_heatmap_recall

        # compute heatmap loss
        loss_heatmap = self.loss_heatmap(
            clip_sigmoid(preds_dict['dense_heatmap']).float(),
            heatmap.float(),
            avg_factor=max(heatmap.eq(1).float().sum().item(), 1),
        )
        loss_dict['loss_heatmap'] = loss_heatmap

        # compute loss 
        layer_score = preds_dict['heatmap']
        
        cls_score = layer_score.permute(0, 2, 1).reshape(
            -1, self.num_classes)
        loss_cls = self.loss_cls(
            cls_score.float(),
            labels.reshape(-1),
            label_weights.reshape(-1),
            avg_factor=max(num_pos, 1),
        )

        center = preds_dict['center']
        height = preds_dict['height']
        rot = preds_dict['rot']
        dim = preds_dict['dim']
        preds = torch.cat(
            [center, height, rot, dim],
            dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
        if 'vel' in preds_dict.keys():
            vel = preds_dict['vel']
            preds = torch.cat([
                center, height, rot, dim, vel
            ], dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
        code_weights = self.train_cfg.get('code_weights', None)
        reg_weights = bbox_weights * bbox_weights.new_tensor(  # noqa: E501
            code_weights )
        loss_bbox = self.loss_bbox(
            preds,
            bbox_targets,
            reg_weights,
            avg_factor=max(num_pos, 1))
        
        loss_dict['loss_cls'] = loss_cls
        loss_dict['loss_bbox'] = loss_bbox
        loss_dict['matched_ious'] = loss_cls.new_tensor(matched_ious)

        return loss_dict
    
    def predict(self, batch_feats, batch_input_metas):
        preds_dicts = self(batch_feats, batch_input_metas)
        res = self.predict_by_feat(preds_dicts, batch_input_metas)
        return res
    
    def predict_by_feat(self,
                        preds_dicts,
                        metas,
                        img=None,
                        rescale=False,
                        for_roi=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
        Returns:
            list[list[dict]]: Decoded bbox, scores and labels for each layer
            & each batch.
        """
        rets = []
        for layer_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_score = preds_dict[0]['heatmap'][
                ..., -self.num_proposals:].sigmoid()
            # if self.loss_iou.loss_weight != 0:
            #    batch_score = torch.sqrt(batch_score * preds_dict[0]['iou'][..., -self.num_proposals:].sigmoid()) # noqa: E501
            #one_hot = F.one_hot(
            #    self.query_labels,
            #    num_classes=self.num_classes).permute(0, 2, 1)
            batch_score = batch_score * preds_dict[0][
                'query_heatmap_score'] #* one_hot

            batch_center = preds_dict[0]['center'][..., -self.num_proposals:]
            batch_height = preds_dict[0]['height'][..., -self.num_proposals:]
            batch_dim = preds_dict[0]['dim'][..., -self.num_proposals:]
            batch_rot = preds_dict[0]['rot'][..., -self.num_proposals:]
            batch_vel = None
            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel'][..., -self.num_proposals:]

            temp = self.bbox_coder.decode(
                batch_score,
                batch_rot,
                batch_dim,
                batch_center,
                batch_height,
                batch_vel,
                filter=True,
            )

            if self.test_cfg['dataset'] == 'nuScenes':
                self.tasks = [
                    dict(
                        num_class=8,
                        class_names=[],
                        indices=[0, 1, 2, 3, 4, 5, 6, 7],
                        radius=-1,
                    ),
                    dict(
                        num_class=1,
                        class_names=['pedestrian'],
                        indices=[8],
                        radius=0.175,
                    ),
                    dict(
                        num_class=1,
                        class_names=['traffic_cone'],
                        indices=[9],
                        radius=0.175,
                    ),
                ]
            elif self.test_cfg['dataset'] == 'Waymo':
                self.tasks = [
                    dict(
                        num_class=1,
                        class_names=['Car'],
                        indices=[0],
                        radius=0.7),
                    dict(
                        num_class=1,
                        class_names=['Pedestrian'],
                        indices=[1],
                        radius=0.7),
                    dict(
                        num_class=1,
                        class_names=['Cyclist'],
                        indices=[2],
                        radius=0.7),
                ]

            ret_layer = []
            for i in range(batch_size):
                boxes3d = temp[i]['bboxes']
                scores = temp[i]['scores']
                labels = temp[i]['labels']
                # adopt circle nms for different categories
                if self.test_cfg['nms_type'] is not None:
                    keep_mask = torch.zeros_like(scores)
                    for task in self.tasks:
                        task_mask = torch.zeros_like(scores)
                        for cls_idx in task['indices']:
                            task_mask += labels == cls_idx
                        task_mask = task_mask.bool()
                        if task['radius'] > 0:
                            if self.test_cfg['nms_type'] == 'circle':
                                boxes_for_nms = torch.cat(
                                    [
                                        boxes3d[task_mask][:, :2],
                                        scores[:, None][task_mask],
                                    ],
                                    dim=1,
                                )
                                task_keep_indices = torch.tensor(
                                    circle_nms(
                                        boxes_for_nms.detach().cpu().numpy(),
                                        task['radius'],
                                    ))
                            else:
                                boxes_for_nms = xywhr2xyxyr(
                                    metas[i]['box_type_3d'](
                                        boxes3d[task_mask][:, :7], 7).bev)
                                top_scores = scores[task_mask]
                                task_keep_indices = nms_bev(
                                    boxes_for_nms,
                                    top_scores,
                                    thresh=task['radius'],
                                    pre_maxsize=self.test_cfg['pre_maxsize'],
                                    post_max_size=self.
                                    test_cfg['post_maxsize'],
                                )
                        else:
                            task_keep_indices = torch.arange(task_mask.sum())
                        if task_keep_indices.shape[0] != 0:
                            keep_indices = torch.where(
                                task_mask != 0)[0][task_keep_indices]
                            keep_mask[keep_indices] = 1
                    keep_mask = keep_mask.bool()
                    ret = dict(
                        bboxes=boxes3d[keep_mask],
                        scores=scores[keep_mask],
                        labels=labels[keep_mask],
                    )
                else:  # no nms
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)

                temp_instances = InstanceData()
                temp_instances.bboxes_3d = metas[0]['box_type_3d'](
                    ret['bboxes'], box_dim=ret['bboxes'].shape[-1])
                temp_instances.scores_3d = ret['scores']
                temp_instances.labels_3d = ret['labels'].int()

                ret_layer.append(temp_instances)

            rets.append(ret_layer)
        assert len(
            rets
        ) == 1, f'only support one layer now, but get {len(rets)} layers'

        return rets[0]
