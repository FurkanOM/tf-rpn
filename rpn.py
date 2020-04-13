import os
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model, Sequential
import numpy as np
import helpers

def generate_base_anchors(hyper_params):
    """Generating top left anchors for given anchor_ratios, anchor_scales and stride values.
    inputs:
        hyper_params = dictionary

    outputs:
        base_anchors = (anchor_count, [y1, x1, y2, x2])
            these values not normalized yet
    """
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
    """Broadcasting base_anchors and generating all anchors for given image parameters.
    inputs:
        img_params = (image height, image width, image output height, image output width)
            these output values need to be calculated for used backbone,
            for VGG16 output dimensions = dimension (height or width) // stride
        hyper_params = dictionary

    outputs:
        anchors = (output_width * output_height * anchor_count, [y1, x1, y2, x2])
            these values in normalized format between [0, 1]
    """
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
    anchors = np.clip(anchors, 0, 1)
    return anchors

def generator(dataset, hyper_params):
    """Tensorflow data generator for fit method, yielding inputs and outputs.
    inputs:
        dataset = tf.data.Dataset, PaddedBatchDataset
        hyper_params = dictionary

    outputs:
        yield inputs, outputs
    """
    while True:
        for image_data in dataset:
            input_img, bbox_deltas, bbox_labels, _ = get_step_data(image_data, hyper_params)
            yield input_img, (bbox_deltas, bbox_labels)

def get_step_data(image_data, hyper_params, mode="training"):
    """Generating one step data for training or inference.
    Batch operations supported.
    inputs:
        image_data =
            img (batch_size, height, width, channels)
            gt_boxes (batch_size, gt_box_size, [y1, x1, y2, x2])
                these values in normalized format between [0, 1]
            gt_labels (batch_size, gt_box_size)
        hyper_params = dictionary
        mode = "training" or "inference"

    outputs:
        input_img = (batch_size, height, width, channels)
            preprocessed image using preprocess_input
        bbox_deltas = (batch_size, output_height, output_width, anchor_count * [delta_y, delta_x, delta_h, delta_w])
            actual outputs for rpn, generating only training mode
        bbox_labels = (batch_size, output_height, output_width, anchor_count)
            actual outputs for rpn, generating only training mode
        anchors = (batch_size, output_height * output_width * anchor_count, [y1, x1, y2, x2])
    """
    img, gt_boxes, gt_labels = image_data
    batch_size = tf.shape(img)[0]
    input_img = preprocess_input(img)
    input_img = tf.image.convert_image_dtype(input_img, tf.float32)
    stride = hyper_params["stride"]
    anchor_count = hyper_params["anchor_count"]
    total_pos_bboxes = hyper_params["total_pos_bboxes"]
    total_neg_bboxes = hyper_params["total_neg_bboxes"]
    total_bboxes = total_pos_bboxes + total_neg_bboxes
    img_params = helpers.get_image_params(img, stride)
    height, width, output_height, output_width = img_params
    total_anchors = output_height * output_width * anchor_count
    anchors = generate_anchors(img_params, hyper_params)
    # We use same anchors for each batch so we multiplied anchors to the batch size
    anchors = tf.tile(tf.expand_dims(anchors, 0), (batch_size, 1, 1))
    if mode != "training":
        return input_img, anchors
    ################################################################################################################
    # Calculate iou values between each bboxes and ground truth boxes
    iou_map = helpers.generate_iou_map(anchors, gt_boxes)
    # Get max index value for each row
    max_indices_each_gt_box = tf.argmax(iou_map, axis=2, output_type=tf.int32)
    # IoU map has iou values for every gt boxes and we merge these values column wise
    merged_iou_map = tf.reduce_max(iou_map, axis=2)
    # Sorted iou values
    sorted_iou_map = tf.argsort(merged_iou_map, direction="DESCENDING")
    # Sort indices for generating masks
    sorted_map_indices = tf.argsort(sorted_iou_map)
    # Generate pos mask for pos bboxes
    pos_mask = tf.less(sorted_map_indices, total_pos_bboxes)
    # Generate neg mask for neg bboxes
    neg_mask = tf.greater(sorted_map_indices, (total_anchors-1) - total_neg_bboxes)
    # Generate pos and negative labels
    pos_labels = tf.where(pos_mask, tf.ones_like(pos_mask, dtype=tf.float32), tf.constant(-1.0, dtype=tf.float32))
    neg_labels = tf.cast(neg_mask, dtype=tf.float32)
    bbox_labels = tf.add(pos_labels, neg_labels)
    #
    gt_boxes_map = tf.gather(gt_boxes, max_indices_each_gt_box, batch_dims=1)
    # Replace negative bboxes with zeros
    expanded_gt_boxes = tf.where(tf.expand_dims(pos_mask, -1), gt_boxes_map, tf.zeros_like(gt_boxes_map))
    # Calculate delta values between anchors and ground truth bboxes
    bbox_deltas = helpers.get_deltas_from_bboxes(anchors, expanded_gt_boxes)
    #
    bbox_deltas = tf.reshape(bbox_deltas, (batch_size, output_height, output_width, anchor_count * 4))
    bbox_labels = tf.reshape(bbox_labels, (batch_size, output_height, output_width, anchor_count))
    #
    return input_img, bbox_deltas, bbox_labels, anchors

class RPNModel(Model):
    def __init__(self, stride, anchor_count, **kwargs):
        super().__init__(**kwargs)
        self.stride = stride
        self.anchor_count = anchor_count
        self.base_model = self.get_base_model()
        self.head_layers = self.get_head_layers()

    def get_base_model(self):
        base_model = VGG16(include_top=False)
        if self.stride == 16:
            base_model = Sequential(base_model.layers[:-1])
        return base_model

    def get_head_layers(self):
        anchor_count = self.anchor_count
        return [
            Conv2D(512, (3, 3), activation="relu", padding="same", name="rpn_conv"),
            Conv2D(anchor_count * 4, (1, 1), activation="linear", name="rpn_reg"),
            Conv2D(anchor_count, (1, 1), activation="sigmoid", name="rpn_cls"),
        ]

    def call(self, x):
        for layer in self.base_model.layers:
            x = layer(x)
        x = self.head_layers[0](x)
        reg_output = self.head_layers[1](x)
        cls_output = self.head_layers[2](x)
        return reg_output, cls_output
