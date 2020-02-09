import os
import argparse
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

###########################################
## Pascal VOC
###########################################
VOC = {
    "max_height": 500,
    "max_width": 500,
}

def get_VOC_data(split):
    assert split in ["train", "validation", "test"]
    dataset, info = tfds.load("voc", split=split, with_info=True)
    class_len = info.features["labels"].num_classes
    data_len = info.splits[split].num_examples
    return dataset, data_len, class_len

def get_image_boundaries(height, width):
    return {
        "top": 0,
        "left": 0,
        "right": width,
        "bottom": height
    }

def update_image_boundaries_with_padding(img_boundaries, padding):
    img_boundaries["top"] = padding["top"]
    img_boundaries["left"] = padding["left"]
    img_boundaries["bottom"] += padding["top"]
    img_boundaries["right"] += padding["left"]
    return img_boundaries

def calculate_iou(bboxes, gt):
    ### Ground truth box normalized y1, x1, y2, x2
    gt_y1, gt_x1, gt_y2, gt_x2 = gt
    gt_width = gt_x2 - gt_x1
    gt_height = gt_y2 - gt_y1
    gt_area = gt_width * gt_height
    ### bbox normalized y1, x1, y2, x2
    bbox_y1, bbox_x1, bbox_y2, bbox_x2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    bbox_width = bbox_x2 - bbox_x1
    bbox_height = bbox_y2 - bbox_y1
    bbox_area = bbox_width * bbox_height
    ### Possible intersection
    x_top = np.maximum(gt_x1, bbox_x1)
    y_top = np.maximum(gt_y1, bbox_y1)
    x_bottom = np.minimum(gt_x2, bbox_x2)
    y_bottom = np.minimum(gt_y2, bbox_y2)
    ### Calculate intersection area
    intersection_area = np.maximum(x_bottom - x_top, 0) * np.maximum(y_bottom - y_top, 0)
    ### Calculate union area
    union_area = gt_area + bbox_area - intersection_area
    # Intersection over Union
    return intersection_area / union_area

def generate_iou_map(bboxes, gt_boxes):
    bbox_count = bboxes.shape[0]
    gt_box_count = gt_boxes.shape[0]
    iou_map = np.zeros((bbox_count, gt_box_count), dtype=np.float32)
    for gt_index, gt_box in enumerate(gt_boxes):
        iou = calculate_iou(bboxes, gt_box)
        iou_map[:, gt_index] = iou
    return iou_map

def get_positive_and_negative_bbox_indices(bboxes, gt_boxes, total_pos_bbox_number=64):
    iou_map = generate_iou_map(bboxes, gt_boxes)
    # any time => iou_map.reshape(output_height, output_width, anchor_count, gt_boxes.shape[0])
    ################################################################
    max_indices_each_gt_box = iou_map.argmax(axis=1)
    # IoU map has iou values for every gt boxes and we merge these values column wise
    merged_iou_map = np.max(iou_map, axis=1)
    masked_merged_iou_map = np.ma.array(merged_iou_map, mask=False)
    # Get max index for every column / gt boxes
    max_indices = iou_map.argmax(axis=0)
    max_indices_each_gt_box[max_indices] = np.arange(max_indices.shape[0])
    # We mask max indices and therefore these values placed front of the sorted maps
    masked_merged_iou_map.mask[max_indices] = True
    sorted_iou_map = masked_merged_iou_map.argsort()[::-1]
    sorted_bbox_indices = sorted_iou_map[:total_pos_bbox_number]
    # We finalize pos anchors with max n anchors
    pos_bbox_indices = np.array((sorted_bbox_indices, max_indices_each_gt_box[sorted_bbox_indices]), dtype=np.int32).transpose()
    ##########
    neg_bbox_indices = np.where(masked_merged_iou_map < 0.3)[0]
    neg_bbox_indices = neg_bbox_indices[~np.isin(neg_bbox_indices, pos_bbox_indices[:,0])]
    neg_bbox_indices_count = neg_bbox_indices.shape[0]
    pos_bbox_indices_count = pos_bbox_indices[:,0].shape[0]
    # If there are more negative anchors than positive
    # randomly select negative anchors as many as positive anchor number
    if neg_bbox_indices_count > pos_bbox_indices_count:
        neg_bbox_indices = np.random.choice(neg_bbox_indices, size=pos_bbox_indices_count, replace=False)
    #
    return pos_bbox_indices, neg_bbox_indices

def get_deltas_from_bboxes(bboxes, gt_boxes, pos_bbox_indices):
    bbox_deltas = np.zeros(bboxes.shape, dtype=np.float32)
    bbox_indices, gt_indices = pos_bbox_indices[:, 0], pos_bbox_indices[:, 1]
    #
    bbox_width = bboxes[bbox_indices, 3] - bboxes[bbox_indices, 1] + 1e-7
    bbox_height = bboxes[bbox_indices, 2] - bboxes[bbox_indices, 0] + 1e-7
    bbox_ctr_x = bboxes[bbox_indices, 1] + 0.5 * bbox_width
    bbox_ctr_y = bboxes[bbox_indices, 0] + 0.5 * bbox_height
    #
    gt_width = gt_boxes[gt_indices, 3] - gt_boxes[gt_indices, 1]
    gt_height = gt_boxes[gt_indices, 2] - gt_boxes[gt_indices, 0]
    gt_ctr_x = gt_boxes[gt_indices, 1] + 0.5 * gt_width
    gt_ctr_y = gt_boxes[gt_indices, 0] + 0.5 * gt_height
    #
    delta_x = (gt_ctr_x - bbox_ctr_x) / bbox_width
    delta_y = (gt_ctr_y - bbox_ctr_y) / bbox_height
    delta_w = np.log(gt_width / bbox_width)
    delta_h = np.log(gt_height / bbox_height)
    #
    bbox_deltas[bbox_indices, 0] = delta_y
    bbox_deltas[bbox_indices, 1] = delta_x
    bbox_deltas[bbox_indices, 2] = delta_h
    bbox_deltas[bbox_indices, 3] = delta_w
    #
    return bbox_deltas

def non_max_suppression(pred_bboxes, pred_labels, top_n_boxes=300):
    nms_indices = tf.image.non_max_suppression(pred_bboxes, pred_labels, top_n_boxes)
    nms_bboxes = tf.gather(pred_bboxes, nms_indices)
    return nms_bboxes.numpy()

def get_bboxes_from_deltas(anchors, deltas):
    bboxes = np.zeros(anchors.shape, dtype=np.float32)
    #
    all_anc_width = anchors[:, 3] - anchors[:, 1]
    all_anc_height = anchors[:, 2] - anchors[:, 0]
    all_anc_ctr_x = anchors[:, 1] + 0.5 * all_anc_width
    all_anc_ctr_y = anchors[:, 0] + 0.5 * all_anc_height
    #
    all_bbox_width = tf.exp(deltas[:, 3]) * all_anc_width
    all_bbox_height = tf.exp(deltas[:, 2]) * all_anc_height
    all_bbox_ctr_x = (deltas[:, 1] * all_anc_width) + all_anc_ctr_x
    all_bbox_ctr_y = (deltas[:, 0] * all_anc_height) + all_anc_ctr_y
    #
    bboxes[:, 0] = all_bbox_ctr_y - (0.5 * all_bbox_height)
    bboxes[:, 1] = all_bbox_ctr_x - (0.5 * all_bbox_width)
    bboxes[:, 2] = all_bbox_height + bboxes[:, 0]
    bboxes[:, 3] = all_bbox_width + bboxes[:, 1]
    #
    return bboxes

def get_predicted_bboxes_and_labels(anchor_count, anchors, pred_bbox_deltas, pred_labels):
    _, output_height, output_width, _ = pred_bbox_deltas.shape
    n_row = output_height * output_width * anchor_count
    pred_bbox_deltas = tf.reshape(pred_bbox_deltas, (n_row, 4))
    pred_labels = tf.reshape(pred_labels, (n_row, ))
    pred_bboxes = get_bboxes_from_deltas(anchors, pred_bbox_deltas)
    return pred_bboxes, pred_labels

def get_input_img(img, input_processor):
    processed_img = input_processor(img)
    return np.expand_dims(processed_img, axis=0)

def get_image_params(img, stride):
    height, width, _ = img.shape
    output_height, output_width = height // stride, width // stride
    return height, width, output_height, output_width

def normalize_bboxes(bboxes, height, width):
    new_bboxes = np.zeros(bboxes.shape, dtype=np.float32)
    new_bboxes[:, 0] = bboxes[:, 0] / height
    new_bboxes[:, 1] = bboxes[:, 1] / width
    new_bboxes[:, 2] = bboxes[:, 2] / height
    new_bboxes[:, 3] = bboxes[:, 3] / width
    return new_bboxes

def update_gt_boxes(gt_boxes, img_height, img_width, padding):
    padded_height = img_height + padding["top"] + padding["bottom"]
    padded_width = img_width + padding["left"] + padding["right"]
    gt_boxes[:, 0] = (np.round(gt_boxes[:, 0] * img_height) + padding["top"]) / padded_height
    gt_boxes[:, 1] = (np.round(gt_boxes[:, 1] * img_width) + padding["left"]) / padded_width
    gt_boxes[:, 2] = (np.round(gt_boxes[:, 2] * img_height) + padding["top"]) / padded_height
    gt_boxes[:, 3] = (np.round(gt_boxes[:, 3] * img_width) + padding["left"]) / padded_width
    return gt_boxes

def img_from_array(array):
    return Image.fromarray(array)

def array_from_img(image):
    return np.array(image)

def draw_grid_map(img, grid_map, stride):
    image = img_from_array(img)
    draw = ImageDraw.Draw(image)
    counter = 0
    for grid in grid_map:
        draw.rectangle((
            grid[0] + stride // 2 - 2,
            grid[1] + stride // 2 - 2,
            grid[2] + stride // 2 + 2,
            grid[3] + stride // 2 + 2), fill=(255, 255, 255, 0))
        counter += 1
    plt.figure()
    plt.imshow(image)
    plt.show()

def draw_bboxes(img, bboxes):
    img_float32 = tf.image.convert_image_dtype(img, dtype=tf.float32)
    colors = tf.cast(np.array([[1, 0, 0, 1]] * 10), dtype=tf.float32)
    img_with_bounding_boxes = tf.image.draw_bounding_boxes(
        np.expand_dims(img_float32, axis=0),
        np.expand_dims(bboxes, axis=0),
        colors
    )
    plt.figure()
    plt.imshow(img_with_bounding_boxes[0])
    plt.show()

def resize_image(image, max_allowed_size):
    width, height = image.size
    max_image_size = max(height, width)
    if max_allowed_size < max_image_size:
        if height > width:
            new_height = max_allowed_size
            new_width = int(round(new_height * (width / height)))
        else:
            new_width = max_allowed_size
            new_height = int(round(new_width * (height / width)))
        image = image.resize((new_width, new_height), Image.ANTIALIAS)
    return image

def get_padding(img_height, img_width, max_height, max_width):
    assert img_height <= max_height
    assert img_width <= max_width
    padding_height = max_height - img_height
    padding_width = max_width - img_width
    top = padding_height // 2
    bottom = padding_height - top
    left = padding_width // 2
    right = padding_width - left
    return {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
    }

# img param => numpy array
def get_padded_img(img, max_height, max_width):
    height, width, _ = img.shape
    padding = get_padding(height, width, max_height, max_width)
    return np.pad(img, ((padding["top"], padding["bottom"]), (padding["left"], padding["right"]), (0,0)), mode="constant"), padding

# It take images as numpy arrays and return max height, max width values
def calculate_max_height_width(imgs):
    h_w_map = np.zeros((len(imgs), 2), dtype=np.int32)
    for index, img in enumerate(imgs):
        h_w_map[index, 0], h_w_map[index, 1], _ = img.shape
    max_val = h_w_map.argmax(axis=0)
    max_height, max_width = h_w_map[max_val[0], 0], h_w_map[max_val[1], 1]
    return max_height, max_width

def handle_args():
    parser = argparse.ArgumentParser(description="Region Proposal Network Implementation")
    parser.add_argument("-handle-gpu", action="store_true", help="Tensorflow 2 GPU compatibility flag")
    args = parser.parse_args()
    return args

def handle_gpu_compatibility():
    # For tf2 GPU compatibility
    try:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(e)
