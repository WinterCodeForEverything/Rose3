from typing import Dict, List, Optional, Tuple
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
import torch.distributed as dist
from mmengine.utils import is_list_of

from mmdet3d.models import MVXTwoStageDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptMultiConfig, OptSampleList

from .grid_mask import GridMask
from .ops import SPConvVoxelization



@MODELS.register_module()
class MFDetector(MVXTwoStageDetector):
    def __init__(
        self,
        use_grid_mask=False,
        **kwargs,
    ) -> None:
        pts_voxel_cfg = kwargs.get('pts_voxel_layer', None)
        kwargs.pop('pts_voxel_layer')
        super(MFDetector, self).__init__(**kwargs)

        self.use_grid_mask = use_grid_mask
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        if pts_voxel_cfg:
            self.pts_voxel_layer = SPConvVoxelization(**pts_voxel_cfg)

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    def init_weights(self) -> None:
        """Initialize model weights."""
        super(MFDetector, self).init_weights()

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if isinstance(img, list):
            img = torch.stack(img, dim=0)
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img.float())
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    def extract_pts_feat(self, points):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        if points is None:
            return None
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                )
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x
    
    @torch.no_grad()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            print( res_voxels.shape )
            print( res_coors.shape )
            print( res_num_points.shape )
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def extract_feat(self, batch_inputs_dict: dict,
                     batch_input_metas: List[dict]) -> tuple:
        """Extract features from images and points.

        Args:
            batch_inputs_dict (dict): Dict of batch inputs. It
                contains

                - points (List[tensor]):  Point cloud of multiple inputs.
                - imgs (tensor): Image tensor with shape (B, C, H, W).
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.
        Returns:
             tuple: Two elements in tuple arrange as
             image features and point cloud features.
        """
        imgs = batch_inputs_dict.get('img', None)
        points = batch_inputs_dict.get('points', None)
        print( len( points ) )
        print( points[0].shape )
        print( points[1].shape )
        img_feats = self.extract_img_feat(imgs, batch_input_metas)
        #print(len(img_feats))
        #print( img_feats[0].shape )
        pts_feats = self.extract_pts_feat(points)
        #print(len(pts_feats))
        #print( pts_feats[0].shape )
        return ( points, pts_feats, img_feats )

    def loss(self, batch_inputs_dict: Dict[List, torch.Tensor],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        """
        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' and `imgs` keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor): Tensor of batch images, has shape
                  (B, C, H ,W)
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, .

        Returns:
            dict[str, Tensor]: A dictionary of loss components.

        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        points, pts_feats, img_feats = self.extract_feat(batch_inputs_dict,
                                                 batch_input_metas)
        losses = self.pts_bbox_head.loss(points, pts_feats, img_feats, batch_data_samples
                                        )
        return losses

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars  # type: ignore
        

        
