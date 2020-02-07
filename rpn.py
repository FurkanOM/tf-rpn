import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model
import numpy as np
import Helpers

def cls_loss(y_true, y_pred):
    indices = tf.where(tf.not_equal(y_true, -1))
    target = tf.gather_nd(y_true, indices)
    output = tf.gather_nd(y_pred, indices)
    lf = tf.losses.BinaryCrossentropy()
    return tf.reduce_mean(lf(target, output))

def reg_loss(y_true, y_pred):
    indices = tf.where(tf.not_equal(y_true, 0))
    target = tf.gather_nd(y_true, indices)
    output = tf.gather_nd(y_pred, indices)
    # Same with the smooth l1 loss
    lf = tf.losses.Huber()
    return tf.reduce_mean(lf(target, output))

def get_image_params(img, stride):
    height, width, _ = img.shape
    output_height, output_width = height // stride, width // stride
    return height, width, output_height, output_width

def update_gt_boxes(gt_boxes, img_height, img_width, padding):
    padded_height = img_height + padding["top"] + padding["bottom"]
    padded_width = img_width + padding["left"] + padding["right"]
    gt_boxes[:, 0] = (np.round(gt_boxes[:, 0] * img_height) + padding["top"]) / padded_height
    gt_boxes[:, 1] = (np.round(gt_boxes[:, 1] * img_width) + padding["left"]) / padded_width
    gt_boxes[:, 2] = (np.round(gt_boxes[:, 2] * img_height) + padding["top"]) / padded_height
    gt_boxes[:, 3] = (np.round(gt_boxes[:, 3] * img_width) + padding["left"]) / padded_width
    return gt_boxes

def get_input_img(img, input_processor):
    processed_img = input_processor(img)
    return np.expand_dims(processed_img, axis=0)

def normalize_bboxes(bboxes, height, width):
    new_bboxes = np.zeros(bboxes.shape, dtype=np.float32)
    new_bboxes[:, 0] = bboxes[:, 0] / height
    new_bboxes[:, 1] = bboxes[:, 1] / width
    new_bboxes[:, 2] = bboxes[:, 2] / height
    new_bboxes[:, 3] = bboxes[:, 3] / width
    return new_bboxes

def generate_base_anchors(stride, ratios, scales):
    center = stride // 2
    base_anchors = []
    for scale in scales:
        for ratio in ratios:
            box_area = scale ** 2
            w = round((box_area / ratio) ** 0.5)
            h = round(w * ratio)
            x_min = center - w / 2
            y_min = center - h / 2
            x_max = center + w / 2
            y_max = center + h / 2
            base_anchors.append([y_min, x_min, y_max, x_max])
    return np.array(base_anchors, dtype=np.float32)

def get_anchors(img_params, anchor_ratios, anchor_scales, stride):
    anchor_count = len(anchor_ratios) * len(anchor_scales)
    height, width, output_height, output_width = img_params
    #
    grid_x = np.arange(0, output_width) * stride
    grid_y = np.arange(0, output_height) * stride
    #
    width_padding = (width - output_width * stride) / 2
    height_padding = (height - output_height * stride) / 2
    grid_x = width_padding + grid_x
    grid_y = height_padding + grid_y
    #
    grid_y, grid_x = np.meshgrid(grid_y, grid_x)
    grid_map = np.vstack((grid_y.ravel(), grid_x.ravel(), grid_y.ravel(), grid_x.ravel())).transpose()
    #
    base_anchors = generate_base_anchors(stride, anchor_ratios, anchor_scales)
    #
    output_area = grid_map.shape[0]
    anchors = base_anchors.reshape((1, anchor_count, 4)) + \
              grid_map.reshape((1, output_area, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((output_area * anchor_count, 4)).astype(np.float32)
    anchors = normalize_bboxes(anchors, height, width)
    return anchors

def get_bbox_deltas_and_labels(img_params, anchors, gt_boxes, anchor_count, stride, img_boundaries):
    height, width, output_height, output_width = img_params
    #############################
    # Positive and negative anchors calculation
    #############################
    # Positive and negative anchor numbers are 128 in original paper
    pos_anchors, neg_anchors = Helpers.get_positive_and_negative_anchors(anchors, gt_boxes, total_pos_anchor_number=64)
    #############################
    # Bbox delta calculation
    #############################
    bbox_deltas = Helpers.get_deltas_from_bboxes(anchors, gt_boxes, pos_anchors)
    #############################
    # Label calculation
    #############################
    # labels => 1 object, 0 background, -1 neutral
    labels = -1 * np.ones((anchors.shape[0], ), dtype=np.float32)
    labels[neg_anchors] = 0
    labels[pos_anchors[:,0]] = 1
    ############################################################
    bbox_deltas = bbox_deltas.reshape(output_height, output_width, anchor_count * 4)
    labels = labels.reshape(output_height, output_width, anchor_count)
    return bbox_deltas, labels

def get_input_output(img, bbox_deltas, labels):
    input = img
    outputs = [
        np.expand_dims(bbox_deltas, axis=0),
        np.expand_dims(labels, axis=0)
    ]
    return input, outputs

def generator(data,
              anchor_ratios,
              anchor_scales,
              stride,
              input_processor,
              max_height=None,
              max_width=None,
              apply_padding=False):
    while True:
        for image_data in data:
            input, outputs, _, _ = get_step_data(
                image_data,
                anchor_ratios,
                anchor_scales,
                stride,
                input_processor,
                max_height,
                max_width,
                apply_padding
            )
            yield input, outputs

def get_step_data(image_data, anchor_ratios, anchor_scales, stride, input_processor, max_height, max_width, apply_padding):
    img = image_data["image"].numpy()
    img_height, img_width, _ = img.shape
    img_boundaries = Helpers.get_image_boundaries(img_height, img_width)
    gt_boxes = image_data["objects"]["bbox"].numpy()
    if apply_padding:
        img, padding = Helpers.get_padded_img(img, max_height, max_width)
        gt_boxes = update_gt_boxes(gt_boxes, img_height, img_width, padding)
        img_boundaries = Helpers.update_image_boundaries_with_padding(img_boundaries, padding)
    img_params = get_image_params(img, stride)
    anchors = get_anchors(img_params, anchor_ratios, anchor_scales, stride)
    anchor_count = len(anchor_ratios) * len(anchor_scales)
    actual_bbox_deltas, actual_labels = get_bbox_deltas_and_labels(img_params, anchors, gt_boxes, anchor_count, stride, img_boundaries)
    input_img = get_input_img(img, input_processor)
    input, outputs = get_input_output(input_img, actual_bbox_deltas, actual_labels)
    return input, outputs, anchors, gt_boxes

def get_model(base_model, anchor_count):
    output = Conv2D(512, (3, 3), activation="relu", padding="same", name="rpn_conv")(base_model.output)
    rpn_cls_output = Conv2D(anchor_count, (1, 1), activation="sigmoid", name="rpn_cls")(output)
    rpn_reg_output = Conv2D(anchor_count * 4, (1, 1), activation="linear", name="rpn_reg")(output)
    rpn_model = Model(inputs=base_model.input, outputs=[rpn_reg_output, rpn_cls_output])
    return rpn_model

def get_model_path(stride):
    main_path = "models"
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    model_path = os.path.join(main_path, "stride_" + str(stride) + "_rpn_model_weights.h5")
    return model_path
