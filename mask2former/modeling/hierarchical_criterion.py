# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list

def boundary_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    inputs = inputs.sigmoid()
    loss = torch.einsum("pd,pd->pd", inputs, targets)
    return loss.mean()

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
        focal: bool = True
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    if focal:
        alpha = 0.25
        gamma = 2.0
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
        loss = alpha_t * (1 - pt) ** gamma * BCE_loss
    else:
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)

def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, use_focal_loss, use_boundary_loss):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        # loss options
        self.boundary_loss = use_boundary_loss
        self.focal_loss = use_focal_loss
    
    def get_weights(self):
        return self.weight_dict

    def update_weights(self, weight_dict):
        self.weight_dict = weight_dict
        print('Boundary Weight Updated: ', self.weight_dict)

    def loss_labels(self, outputs, targets, indices, num_masks, level):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = outputs[f"{level}_pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[f"{level}_labels"][J] for t, (_, J) in zip(targets, indices)])
        if level == "plant":
            target_classes = torch.full(
                src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
            )
            target_classes[idx] = target_classes_o
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes)
        else:
            target_classes = torch.full(
                src_logits.shape[:2], self.num_classes - 1, dtype=torch.int64, device=src_logits.device
            )
            target_classes[idx] = target_classes_o
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes)
        losses = {f"{level}_loss_ce": loss_ce}
        return losses
    
    def loss_masks(self, outputs, targets, indices, num_masks, level):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """

        losses = {}
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs[f"{level}_pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t[f"{level}_masks"] for t in targets]
        dist_maps = [t[f"{level}_dist_maps"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        
        dist_maps, valid = nested_tensor_from_tensor_list(dist_maps).decompose()
        dist_maps = dist_maps[tgt_idx]

        dist_maps = dist_maps.to(dtype=torch.float16)

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]
        dist_maps = dist_maps[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
                mode='nearest'
            ).squeeze(1)

            # get gt labels
            point_boundaries = point_sample(
                dist_maps,
                point_coords,
                align_corners=False,
                mode='nearest'
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)
        
        if self.boundary_loss:
            losses[f"{level}_loss_boundary"] = boundary_loss(point_logits, point_boundaries, num_masks)
        losses[f"{level}_loss_mask"] = sigmoid_ce_loss_jit(point_logits, point_labels, num_masks, self.focal_loss)
        losses[f"{level}_loss_dice"] = dice_loss_jit(point_logits, point_labels, num_masks)
        del dist_maps
        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_num_masks(self, level, outputs, targets):
        num_masks = sum(len(t[f"{level}_masks"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()
        return num_masks

    def get_loss(self, loss, outputs, targets):
        plant_matcher = self.matcher(outputs, targets, "plant")
        leaf_matcher = self.matcher(outputs, targets, "leaf")
        plant_num_masks = self.get_num_masks("plant", outputs, targets)
        leaf_num_masks = self.get_num_masks("leaf", outputs, targets)
        loss_map = {
            "plant_labels": self.loss_labels(outputs, targets, plant_matcher, plant_num_masks, "plant"),
            "leaf_labels": self.loss_labels(outputs, targets, leaf_matcher, leaf_num_masks, "leaf"),
            "plant_masks": self.loss_masks(outputs, targets, plant_matcher, plant_num_masks, "plant"),
            "leaf_masks": self.loss_masks(outputs, targets, leaf_matcher, leaf_num_masks, "leaf")
        }
        return loss_map[loss]

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))

        for i, aux_outputs in enumerate(outputs["aux_outputs"]):
            for loss in self.losses:
                l_dict = self.get_loss(loss, aux_outputs, targets)
                l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
