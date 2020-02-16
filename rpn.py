import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model
import numpy as np
import helpers

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

def generate_base_anchors(hyper_params):
    stride = hyper_params["stride"]
    anchor_ratios = hyper_params["anchor_ratios"]
    anchor_scales = hyper_params["anchor_scales"]
    center = stride // 2
    base_anchors = []
    for scale in anchor_scales:
        for ratio in anchor_ratios:
            box_area = scale ** 2
            w = round((box_area / ratio) ** 0.5)
            h = round(w * ratio)
            x_min = center - w / 2
            y_min = center - h / 2
            x_max = center + w / 2
            y_max = center + h / 2
            base_anchors.append([y_min, x_min, y_max, x_max])
    return np.array(base_anchors, dtype=np.float32)

def generate_anchors(img_params, hyper_params):
    anchor_count = hyper_params["anchor_count"]
    stride = hyper_params["stride"]
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
    base_anchors = generate_base_anchors(hyper_params)
    #
    output_area = grid_map.shape[0]
    anchors = base_anchors.reshape((1, anchor_count, 4)) + \
              grid_map.reshape((1, output_area, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((output_area * anchor_count, 4)).astype(np.float32)
    anchors = helpers.normalize_bboxes(anchors, height, width)
    return anchors

def get_bbox_deltas_and_labels(anchors, gt_boxes, hyper_params, img_params):
    anchor_count = hyper_params["anchor_count"]
    height, width, output_height, output_width = img_params
    #############################
    # Positive and negative anchors calculation
    #############################
    # Positive and negative anchor numbers are 128 in original paper
    pos_bbox_indices, neg_bbox_indices, gt_box_indices = helpers.get_selected_indices([anchors, gt_boxes, hyper_params["total_pos_bboxes"]])
    #############################
    # Bbox delta calculation
    #############################
    pos_gt_boxes_map = tf.gather(gt_boxes, gt_box_indices)
    final_gt_boxes = tf.scatter_nd(tf.expand_dims(pos_bbox_indices, 1), pos_gt_boxes_map, tf.shape(anchors))
    bbox_deltas = helpers.get_deltas_from_bboxes(anchors, final_gt_boxes)
    #############################
    # Label calculation
    #############################
    # labels => 1 object, 0 background, -1 neutral
    labels = -1 * np.ones((anchors.shape[0], ), dtype=np.float32)
    labels[neg_bbox_indices] = 0
    labels[pos_bbox_indices] = 1
    ############################################################
    bbox_deltas = tf.reshape(bbox_deltas, (output_height, output_width, anchor_count * 4))
    labels = tf.reshape(labels, (output_height, output_width, anchor_count))
    return bbox_deltas, labels

def generator(data, hyper_params, input_processor):
    while True:
        for image_data in data:
            input_img, img_params, gt_boxes, _ = helpers.preprocessing(image_data, hyper_params, input_processor)
            input, outputs, _ = get_step_data(input_img, img_params, gt_boxes, hyper_params)
            yield input, outputs

def get_input_output(input_img, bbox_deltas, labels):
    input = input_img
    outputs = [
        np.expand_dims(bbox_deltas, axis=0),
        np.expand_dims(labels, axis=0)
    ]
    return input, outputs

def get_step_data(input_img, img_params, gt_boxes, hyper_params):
    anchors = generate_anchors(img_params, hyper_params)
    actual_bbox_deltas, actual_labels = get_bbox_deltas_and_labels(anchors, gt_boxes, hyper_params, img_params)
    input, outputs = get_input_output(input_img, actual_bbox_deltas, actual_labels)
    return input, outputs, anchors

def get_model(base_model, hyper_params):
    output = Conv2D(512, (3, 3), activation="relu", padding="same", name="rpn_conv")(base_model.output)
    rpn_cls_output = Conv2D(hyper_params["anchor_count"], (1, 1), activation="sigmoid", name="rpn_cls")(output)
    rpn_reg_output = Conv2D(hyper_params["anchor_count"] * 4, (1, 1), activation="linear", name="rpn_reg")(output)
    rpn_model = Model(inputs=base_model.input, outputs=[rpn_reg_output, rpn_cls_output])
    return rpn_model

def get_model_path(stride):
    main_path = "models"
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    model_path = os.path.join(main_path, "stride_{0}_rpn_model_weights.h5".format(stride))
    return model_path
