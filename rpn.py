import tensorflow as tf
import numpy as np
import math
import Helpers

def generate_base_anchors(stride, ratios, scales):
    center = stride // 2
    base_anchors = []
    for scale in scales:
        for ratio in ratios:
            box_area = scale ** 2
            w = round(math.sqrt(box_area / ratio))
            h = round(w * ratio)
            x_min = center - w / 2
            y_min = center - h / 2
            x_max = center + w / 2
            y_max = center + h / 2
            base_anchors.append([x_min, y_min, x_max, y_max])
    return np.array(base_anchors)

def calculate_iou(anc, gt):
    ### Ground truth box x1, y1, x2, y2
    gt_x1, gt_y1, gt_x2, gt_y2 = gt
    gt_width = gt_x2 - gt_x1
    gt_height = gt_y2 - gt_y1
    gt_area = gt_width * gt_height
    ### Anchor x1, y1, x2, y2
    anc_x1, anc_y1, anc_x2, anc_y2 = anc
    anc_width = anc_x2 - anc_x1
    anc_height = anc_y2 - anc_y1
    anc_area = anc_width * anc_height
    ### Possible intersection
    x_top = max(gt_x1, anc_x1)
    y_top = max(gt_y1, anc_y1)
    x_bottom = min(gt_x2, anc_x2)
    y_bottom = min(gt_y2, anc_y2)
    ### Check intersection
    if x_bottom < x_top or y_bottom < y_top:
        return 0.0
    ### Calculate intersection area
    intersection_area = (x_bottom - x_top) * (y_bottom - y_top)
    ### Calculate union area
    union_area = gt_area + anc_area - intersection_area
    # Intersection over Union
    return intersection_area / union_area

def generate_iou_map(anchors, objects, width, height):
    anchor_count = anchors.shape[0]
    object_count = len(objects)
    iou_map = np.zeros((anchor_count, object_count), dtype=np.float)
    for anc_index, anchor in enumerate(anchors):
        if anchor[0] < 0 or anchor [1] < 0 or anchor[2] > width or anchor[3] > height:
            continue
        for gt_index, obj in enumerate(objects):
            gt_box = obj["bbox"]
            gt = [gt_box["x_min"], gt_box["y_min"], gt_box["x_max"], gt_box["y_max"]]
            iou = calculate_iou(anchor, gt)
            iou_map[anc_index, gt_index] = iou
    return iou_map

def get_bbox_deltas(anchors, objects, pos_anchors):
    bbox_deltas = np.zeros(anchors.shape)
    for pos_anchor in pos_anchors:
        index, obj_n = pos_anchor
        anchor = anchors[index]
        obj = objects[obj_n]
        #
        anc_width = anchor[2] - anchor[0]
        anc_height = anchor[3] - anchor[1]
        anc_ctr_x = anchor[0] + 0.5 * anc_width
        anc_ctr_y = anchor[1] + 0.5 * anc_height
        #
        gt_box = obj["bbox"]
        gt_width = gt_box["x_max"] - gt_box["x_min"]
        gt_height = gt_box["y_max"] - gt_box["y_min"]
        gt_ctr_x = gt_box["x_min"] + 0.5 * gt_width
        gt_ctr_y = gt_box["y_min"] + 0.5 * gt_height
        #
        delta_x = (gt_ctr_x - anc_ctr_x) / anc_width
        delta_y = (gt_ctr_y - anc_ctr_y) / anc_height
        delta_w = np.log(gt_width / anc_width)
        delta_h = np.log(gt_height / anc_height)
        #
        bbox_deltas[index] = [delta_x, delta_y, delta_w, delta_h]
    #
    return bbox_deltas

def rpn_cls_loss(y_true, y_pred):
    indices = tf.where(tf.not_equal(y_true, -1))
    target = tf.gather_nd(y_true, indices)
    output = tf.gather_nd(y_pred, indices)
    lf = tf.losses.BinaryCrossentropy()
    return tf.reduce_mean(lf(target, output))

def rpn_reg_loss(y_true, y_pred):
    indices = tf.where(tf.not_equal(y_true, 0))
    target = tf.gather_nd(y_true, indices)
    output = tf.gather_nd(y_pred, indices)
    # Same with the smooth l1 loss
    lf = tf.losses.Huber()
    return tf.reduce_mean(lf(target, output))

# This method was implemented by examining the python
# implementation of the code in the original article.
def get_rpn_data(img, objects, anchor_ratios, anchor_scales, stride):
    anchor_count = len(anchor_ratios) * len(anchor_scales)
    height, width, _ = img.shape
    output_height, output_width = height // stride, width // stride
    #
    grid_x = np.arange(0, output_width) * stride
    grid_y = np.arange(0, output_height) * stride
    #
    width_padding = (width - output_width * stride) / 2
    height_padding = (height - output_height * stride) / 2
    grid_x = width_padding + grid_x
    grid_y = height_padding + grid_y
    #
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    grid_map = np.vstack((grid_x.ravel(), grid_y.ravel(), grid_x.ravel(), grid_y.ravel())).transpose()
    #
    base_anchors = generate_base_anchors(stride, anchor_ratios, anchor_scales)
    #
    output_area = grid_map.shape[0]
    anchors = base_anchors.reshape((1, anchor_count, 4)) + \
              grid_map.reshape((1, output_area, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((output_area * anchor_count, 4))
    #
    iou_map = generate_iou_map(anchors, objects, width, height)
    # any time => iou_map.reshape(output_height, output_width, anchor_count, len(objects))
    ################################################################
    pos_anchor_indices, pos_gt_box_indices = np.where(iou_map > 0.7)
    gt_boxes_best_iou_indices = iou_map.argmax(axis=0)
    gt_boxes_best_iou_values = iou_map[gt_boxes_best_iou_indices, np.arange(iou_map.shape[1])]
    best_pos_anchor_indices, best_pos_gt_box_indices = np.where(iou_map == gt_boxes_best_iou_values)
    pos_anchor_indices = np.concatenate((pos_anchor_indices, best_pos_anchor_indices))
    pos_gt_box_indices = np.concatenate((pos_gt_box_indices, best_pos_gt_box_indices))
    pos_anchors = np.stack((pos_anchor_indices, pos_gt_box_indices)).transpose()
    pos_anchors = np.unique(pos_anchors, axis=0)
    #
    max_element_column = iou_map.argmax(axis=1)
    merged_iou_map = iou_map[np.arange(iou_map.shape[0]), max_element_column]
    neg_anchors = np.where(merged_iou_map < 0.3)[0]
    #############################
    # Bbox calculation
    #############################
    bbox_deltas = get_bbox_deltas(anchors, objects, pos_anchors)
    #############################
    # Label calculation
    #############################
    # labels => 1 object, 0 background, -1 neutral
    labels = -1 * np.ones((iou_map.shape[0], ), dtype=np.float32)
    labels[neg_anchors] = 0
    labels[pos_anchors[:,0]] = 1
    neg_anchors_count = len(neg_anchors)
    pos_anchors_count = len(pos_anchors[:,0])
    # If there are more negative anchors than positive
    # Randomly change some negative anchors to neutral
    if neg_anchors_count > pos_anchors_count:
        new_neutral_anchors = np.random.choice(neg_anchors, size=(neg_anchors_count - pos_anchors_count), replace=False)
        labels[new_neutral_anchors] = -1
    ############################################################
    bbox_deltas = bbox_deltas.reshape(output_height, output_width, anchor_count * 4)
    bbox_deltas = np.expand_dims(bbox_deltas, axis=0)
    labels = labels.reshape(output_height, output_width, anchor_count)
    labels = np.expand_dims(labels, axis=0)
    return bbox_deltas, labels

def rpn_feed(data, anchor_ratios, anchor_scales, stride, input_processor):
    while True:
        for image_data in data:
            img = Helpers.get_image(image_data["image_path"], as_array=True)
            bbox_deltas, labels = get_rpn_data(img, image_data["objects"], anchor_ratios, anchor_scales, stride)
            img = input_processor(img)
            img = np.expand_dims(img, axis=0)
            yield img, [bbox_deltas, labels]
