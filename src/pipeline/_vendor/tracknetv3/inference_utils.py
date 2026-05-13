"""Inference helper functions extracted from test.py.

test.py has a module-level ``from pycocotools.coco import COCO`` that crashes
unless pycocotools is installed. Since predict.py only needs the three
functions below (none of which use pycocotools), they live here instead.
"""
import math

import cv2
import numpy as np
import torch


def get_ensemble_weight(seq_len, eval_mode):
    """Get weight for temporal ensemble.

    :param seq_len: Length of input sequence.
    :param eval_mode: 'average' for uniform weight, 'weight' for positional.
    :return: Weight tensor of shape ``(seq_len,)``.
    """
    if eval_mode == 'average':
        weight = torch.ones(seq_len) / seq_len
    elif eval_mode == 'weight':
        weight = torch.ones(seq_len)
        for i in range(math.ceil(seq_len / 2)):
            weight[i] = (i + 1)
            weight[seq_len - i - 1] = (i + 1)
        weight = weight / weight.sum()
    else:
        raise ValueError('Invalid mode')

    return weight


def predict_location(heatmap):
    """Get coordinates from a heatmap.

    :param heatmap: Single heatmap array of shape ``(H, W)``.
    :return: ``(x, y, w, h)`` bounding box of the largest response area.
    """
    if np.amax(heatmap) == 0:
        return 0, 0, 0, 0

    (cnts, _) = cv2.findContours(heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in cnts]

    max_area_idx = 0
    max_area = rects[0][2] * rects[0][3]
    for i in range(1, len(rects)):
        area = rects[i][2] * rects[i][3]
        if area > max_area:
            max_area_idx = i
            max_area = area
    x, y, w, h = rects[max_area_idx]

    return x, y, w, h


def generate_inpaint_mask(pred_dict, th_h=30):
    """Generate inpaint mask from predicted trajectory.

    :param pred_dict: Prediction dict with keys ``Frame``, ``X``, ``Y``,
        ``Visibility``.
    :param th_h: Height threshold (pixels) for y coordinate.
    :return: Inpaint mask as a list of 0/1 values.
    """
    y = np.array(pred_dict['Y'])
    vis_pred = np.array(pred_dict['Visibility'])
    inpaint_mask = np.zeros_like(y)
    i = 0  # index where ball starts to disappear
    j = 0  # index where ball starts to appear
    threshold = th_h
    while j < len(vis_pred):
        while i < len(vis_pred) - 1 and vis_pred[i] == 1:
            i += 1
        j = i
        while j < len(vis_pred) - 1 and vis_pred[j] == 0:
            j += 1
        if j == i:
            break
        elif i == 0 and y[j] > threshold:
            inpaint_mask[:j] = 1
        elif (i > 1 and y[i - 1] > threshold) and (j < len(vis_pred) and y[j] > threshold):
            inpaint_mask[i:j] = 1
        else:
            # ball is out of the field of camera view
            pass
        i = j

    return inpaint_mask.tolist()
