import argparse
import math
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
from typing import Dict
from phenobench.evaluation.auxiliary.common import get_png_files_in_dir, load_file_as_tensor, load_file_as_int_tensor


def parse_args() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('--phenobench_dir', required=True, type=Path, help='Path to ground truth directory.')
    parser.add_argument('--prediction_dir', required=True, type=Path, help='Path to prediction directory.')
    parser.add_argument('--split', default='val', type=str, help='Specify which split to use for evaluation.')

    args = vars(parser.parse_args())

    return args

def assign_leaves(crop_instances, leaf_instances):
    results = {}
    for leaf_instance_id in np.unique(leaf_instances):
        if leaf_instance_id == 0:
            continue
        plant_instance_ids, counts = np.unique(crop_instances[leaf_instances == leaf_instance_id], return_counts=True)
    
        plant_instance_id = 0
        max_count = 0
        for value, count in zip(plant_instance_ids, counts):
            if max_count < count:
                plant_instance_id = value
                max_count = count

        leaves = results.get(plant_instance_id, [])
        leaves.append(leaf_instance_id)
        results[plant_instance_id] = leaves

    return results

def evaluate(plant_prediction, plant_groundtruth, leaf_prediction, leaf_groundtruth):
    numerator_iou = torch.tensor(0.)
    tp_labels = []
    class_matches = 0
    unmatched_class_predictions = 0
    leaf_count_true_positive = 0
    tp_leaf_count_se = 0
    pred_leaf_count_se = 0
    act_leaf_count_se = 0

    # take the non-zero INSTANCE labels for both prediction and groundtruth
    labels = torch.unique(plant_prediction)
    gt_labels = torch.unique(plant_groundtruth)
    labels = labels[labels > 0]
    gt_labels = gt_labels[gt_labels > 0]

    gt_masks = [plant_groundtruth == gt_label for gt_label in gt_labels]
    gt_areas = [(plant_groundtruth == gt_label).sum() for gt_label in gt_labels]

    gt_plant_leaf_labels = assign_leaves(plant_groundtruth, leaf_groundtruth)
    pred_plant_leaf_labels = assign_leaves(plant_prediction, leaf_prediction)

    gt_leaf_labels = [gt_plant_leaf_labels.get(int(gt_label), []) for gt_label in gt_labels]
    not_matched = [True] * len(gt_labels)

    for label in labels: # for each predicted label
      iou = torch.tensor(0.)
      best_idx = 0

      pred_mask = (plant_prediction == label)
      pred_area = (plant_prediction == label).sum()
      pred_leaf_count = len(pred_plant_leaf_labels.get(int(label), []))

      for idx in np.where(not_matched)[0]:
        gt_mask = gt_masks[idx]
        gt_area = gt_areas[idx]
        # compute iou with all instance gt labels and store the best
        intersection = ((pred_mask & gt_mask).sum()).float()
        union = (pred_area + gt_area - intersection).float()

        iou_tmp = intersection / union
        if iou_tmp > iou:
          iou = iou_tmp
          best_idx = idx

      # if the best iou is above 0.5, store the match pred_label-gt_label-iou
      if iou > 0.5:
        class_matches += 1
        numerator_iou += iou
        not_matched[best_idx] = False
        gt_leaf_count = len(gt_leaf_labels[best_idx])
        if pred_leaf_count == gt_leaf_count:
            leaf_count_true_positive += 1
        else:
            pred_leaf_count_se += (pred_leaf_count - gt_leaf_count) ** 2
            act_leaf_count_se += (pred_leaf_count - gt_leaf_count) ** 2
            tp_leaf_count_se += (pred_leaf_count - gt_leaf_count) ** 2
      else:
        # unmatched_class_predictions.append(label.item())
        unmatched_class_predictions += 1
        pred_leaf_count_se += (pred_leaf_count - 0) ** 2

    for idx in np.where(not_matched)[0]:
         gt_leaf_count = len(gt_leaf_labels[idx])
         act_leaf_count_se += (0 - gt_leaf_count) ** 2

    plant_true_positives = class_matches # len(class_matches) # TP = number of matches
    # FP = number of unmatched predictions
    plant_false_positives = unmatched_class_predictions # len(unmatched_class_predictions)
    # FN = number of unmatched gt labels
    plant_false_negatives = len(gt_labels) - class_matches #len(class_matches)

    return plant_true_positives, plant_false_positives, plant_false_negatives, leaf_count_true_positive, pred_leaf_count_se, act_leaf_count_se, tp_leaf_count_se, len(gt_labels)

args = parse_args()

# ------- Ground Truth -------
gt_plant_instance_fnames = get_png_files_in_dir(args['phenobench_dir'] / args['split'] / 'plant_instances')
gt_semantic_fnames = get_png_files_in_dir(args['phenobench_dir'] / args['split'] / 'semantics')
gt_leaf_instance_fnames = get_png_files_in_dir(args['phenobench_dir'] / args['split'] / 'leaf_instances')

# ------- Predictions -------
pred_plant_instance_fnames = get_png_files_in_dir(args['prediction_dir'] / 'plant_instances')
pred_semantic_fnames = get_png_files_in_dir(args['prediction_dir'] / 'semantics')
pred_leaf_instance_fnames = get_png_files_in_dir(args['prediction_dir'] / 'leaf_instances')

# ------- Load the mapping from randomized to original fnames -------
pred_instance_original_fnames = pred_plant_instance_fnames
pred_semantic_original_fnames = pred_semantic_fnames
fname_mapping_reverse = {fname: fname for fname in pred_plant_instance_fnames}

# ------- Accumulators -------
total_gt = 0
plant_total_tp = 0
plant_total_fp = 0
plant_total_fn = 0
leaf_total_tp = 0
pred_leaf_total_se = 0
act_leaf_total_se = 0
tp_leaf_total_se = 0

# ------- Calculate correct predicitions -------
n_total = len(gt_plant_instance_fnames)
for gt_plant_instance_fname, gt_semantic_fname, gt_leaf_instance_fname, pred_plant_instance_fname, pred_semantic_fname, pred_leaf_instance_fname in tqdm(zip(
    gt_plant_instance_fnames, gt_semantic_fnames, gt_leaf_instance_fnames, pred_plant_instance_fnames, pred_semantic_fnames, pred_leaf_instance_fnames), total=n_total):

    assert gt_plant_instance_fname == gt_semantic_fname == gt_leaf_instance_fname

    gt_plant_instance_map = load_file_as_tensor(args['phenobench_dir'] / args['split'] / 'plant_instances' /
                                            gt_plant_instance_fname).squeeze()

    gt_leaf_instance_map = load_file_as_tensor(args['phenobench_dir'] / args['split'] / 'leaf_instances' /
                                            gt_leaf_instance_fname).squeeze()

    gt_semantics = load_file_as_tensor(args['phenobench_dir'] / args['split'] / 'semantics' /
                                            gt_semantic_fname).squeeze()

    pred_plant_instance_map = load_file_as_int_tensor(args['prediction_dir'] / 'plant_instances' /
                                            fname_mapping_reverse[pred_plant_instance_fname]).squeeze()
    pred_semantics = load_file_as_int_tensor(args['prediction_dir'] / 'semantics' /
                                            fname_mapping_reverse[pred_semantic_fname]).squeeze().numpy()

    pred_leaf_instance_map = load_file_as_int_tensor(args['prediction_dir'] / 'leaf_instances' /
                                            fname_mapping_reverse[pred_leaf_instance_fname]).squeeze()

    gt_crop_instances = gt_plant_instance_map.clone()
    gt_crop_instances[gt_semantics != 1] = 0

    pred_crop_instances = pred_plant_instance_map.clone()

    pred_crop_instances[pred_semantics != 1] = 0

    plant_tp, plant_fp, plant_fn, leaf_tp, pred_leaf_se, act_leaf_se, tp_leaf_se, gt_img_total = evaluate(pred_crop_instances, gt_crop_instances, pred_leaf_instance_map, gt_leaf_instance_map)

    plant_total_tp += plant_tp
    plant_total_fp += plant_fp
    plant_total_fn += plant_fn
    leaf_total_tp += leaf_tp
    pred_leaf_total_se += pred_leaf_se
    act_leaf_total_se += pred_leaf_se
    total_gt += gt_img_total
    tp_leaf_total_se += tp_leaf_se

print('Plant True Positive:', plant_total_tp)
print('Plant False Positive:', plant_total_fp)
print('Plant False Negative:', plant_total_fn)
print('Leaf Count True Positive:', leaf_total_tp)
print('GT Total', total_gt)
print('TP Leaf Count RMSE:', math.sqrt(tp_leaf_total_se/(plant_total_tp)))
print('Pred Leaf Count RMSE:', math.sqrt(pred_leaf_total_se/(plant_total_tp + plant_total_fp)))
print('Act Leaf Count RMSE:', math.sqrt(act_leaf_total_se/total_gt))



