from __future__ import division
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area =    torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                    torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, num_classes, sl_num, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    # print(prediction.shape)
    slice_num=sl_num
    box_corner = prediction.new(prediction.shape)
    # print(prediction.shape,box_corner.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]
    # print(prediction.shape)

    # box_corner[:, :, 4] = prediction[:, :, 4] - prediction[:, :, 6] / 2
    # box_corner[:, :, 5] = prediction[:, :, 5] - prediction[:, :, 7] / 2
    # box_corner[:, :, 6] = prediction[:, :, 4] + prediction[:, :, 6] / 2
    # box_corner[:, :, 7] = prediction[:, :, 5] + prediction[:, :, 7] / 2
    # prediction[:, :, :8] = box_corner[:, :, :8]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        # conf_mask1 = (image_pred[:, 9] >= conf_thres).squeeze()
        # print(image_pred.shape,conf_mask.shape,conf_mask1.shape)
        # image_pred1 = image_pred[conf_mask1]
        image_pred = image_pred[conf_mask]
        # print(image_pred.shape)

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1,  keepdim=True)
        # print(class_conf.shape,class_pred.shape)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        # print(image_pred[:, :4].shape,image_pred[:, 8].unsqueeze(1).shape,class_conf.shape)
        # detections = torch.cat((image_pred[:, :5], image_pred[:, 8].unsqueeze(1), class_conf.float(), class_pred.float()), 1)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float(), image_pred[:, 6:6+4*slice_num]), 1)#for4pairs give 22 for 6 pairs give 30,for 15 pairs give 66
        # detections = torch.cat((image_pred[:, 4:9], class_conf.float(), class_pred.float()), 1)
        # print(detections.shape)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1-4*slice_num].cpu().unique()#for 4 pairs give 16;for 6 pairs give 24;for 15 pairs give
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1-4*slice_num] == c]#for 4 pairs give 16
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))

    return output
def non_max_suppression_plus(prediction, num_classes, sl_num, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    # print(prediction.shape)
    slice_num=sl_num
    box_corner = prediction.new(prediction.shape)
    # print(prediction.shape,box_corner.shape)
    # box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    # box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    # box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    # box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    # prediction[:, :, :4] = box_corner[:, :, :4]
    # print(prediction.shape)

    # box_corner[:, :, 4] = prediction[:, :, 4] - prediction[:, :, 6] / 2
    # box_corner[:, :, 5] = prediction[:, :, 5] - prediction[:, :, 7] / 2
    # box_corner[:, :, 6] = prediction[:, :, 4] + prediction[:, :, 6] / 2
    # box_corner[:, :, 7] = prediction[:, :, 5] + prediction[:, :, 7] / 2
    # prediction[:, :, :8] = box_corner[:, :, :8]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 2] >= conf_thres).squeeze()
        # conf_mask1 = (image_pred[:, 9] >= conf_thres).squeeze()
        # print(image_pred.shape,conf_mask.shape,conf_mask1.shape)
        # image_pred1 = image_pred[conf_mask1]
        image_pred = image_pred[conf_mask]
        # print(image_pred.shape)

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        # class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1,  keepdim=True)
        # print(class_conf.shape,class_pred.shape)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        # print(image_pred[:, :4].shape,image_pred[:, 8].unsqueeze(1).shape,class_conf.shape)
        # detections = torch.cat((image_pred[:, :5], image_pred[:, 8].unsqueeze(1), class_conf.float(), class_pred.float()), 1)
        detections = torch.cat((image_pred[:, :3], image_pred[:, 3:3+4*slice_num]), 1)#for4pairs give 22 for 6 pairs give 30,for 15 pairs give 66
        # detections = torch.cat((image_pred[:, 4:9], class_conf.float(), class_pred.float()), 1)
        # print(detections.shape)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1-4*slice_num].cpu().unique()#for 4 pairs give 16;for 6 pairs give 24;for 15 pairs give
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1-4*slice_num] == c]#for 4 pairs give 16
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))

    return output
