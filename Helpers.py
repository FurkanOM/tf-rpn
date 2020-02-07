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

def get_image(path, as_array=False):
    image = Image.open(path)
    return array_from_img(image) if as_array else image

def get_image_boundaries(height, width):
    return {
        "top": 0,
        "left": 0,
        "right": width,
        "bottom": height
    }

def calculate_iou(anc, gt):
    ### Ground truth box normalized y1, x1, y2, x2
    gt_y1, gt_x1, gt_y2, gt_x2 = gt
    gt_width = gt_x2 - gt_x1
    gt_height = gt_y2 - gt_y1
    gt_area = gt_width * gt_height
    ### Anchor normalized y1, x1, y2, x2
    anc_y1, anc_x1, anc_y2, anc_x2 = anc
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

def generate_iou_map(anchors, gt_boxes):
    anchor_count = anchors.shape[0]
    gt_box_count = gt_boxes.shape[0]
    iou_map = np.zeros((anchor_count, gt_box_count), dtype=np.float32)
    for anc_index, anchor in enumerate(anchors):
        for gt_index, gt_box_data in enumerate(gt_boxes):
            iou = calculate_iou(anchor, gt_box_data)
            iou_map[anc_index, gt_index] = iou
    return iou_map

def get_positive_and_negative_anchors(anchors, gt_boxes, total_pos_anchor_number=64):
    iou_map = generate_iou_map(anchors, gt_boxes)
    # any time => iou_map.reshape(output_height, output_width, anchor_count, gt_boxes.shape[0])
    ################################################################
    total_gt_box_count = gt_boxes.shape[0]
    max_indices_each_gt_box = iou_map.argmax(axis=1)
    # IoU map has iou values for every gt boxes and we merge these values column wise
    merged_iou_map = iou_map[np.arange(iou_map.shape[0]), max_indices_each_gt_box]
    masked_merged_iou_map = np.ma.array(merged_iou_map, mask=False)
    # First we calculate max overlapped box for every ground truth box
    for n_col in range(total_gt_box_count):
        anchor_indices_for_gt_box = np.where(max_indices_each_gt_box == n_col)[0]
        if anchor_indices_for_gt_box.shape[0] == 0:
            continue
        max_anchor_index_for_gt_box = iou_map[:, n_col].argmax()
        masked_merged_iou_map.mask[max_anchor_index_for_gt_box] = True
    #
    sorted_iou_map = masked_merged_iou_map.argsort()[::-1]
    sorted_anchor_indices = sorted_iou_map[:total_pos_anchor_number]
    # We finalize pos anchors with max n anchors
    pos_anchors = np.array((sorted_anchor_indices, max_indices_each_gt_box[sorted_anchor_indices]), dtype=np.int32).transpose()
    ##########
    neg_anchors = np.where(masked_merged_iou_map < 0.3)[0]
    neg_anchors = neg_anchors[~np.isin(neg_anchors, pos_anchors[:,0])]
    neg_anchors_count = len(neg_anchors)
    pos_anchors_count = len(pos_anchors[:,0])
    # If there are more negative anchors than positive
    # randomly select negative anchors as many as positive anchor number
    if neg_anchors_count > pos_anchors_count:
        neg_anchors = np.random.choice(neg_anchors, size=pos_anchors_count, replace=False)
    #
    return pos_anchors, neg_anchors

def get_deltas_from_bboxes(anchors, gt_boxes, pos_anchors):
    bbox_deltas = np.zeros(anchors.shape, dtype=np.float32)
    anchor_indices, gt_indices = pos_anchors[:, 0], pos_anchors[:, 1]
    #
    anc_width = anchors[anchor_indices, 3] - anchors[anchor_indices, 1] + 1e-7
    anc_height = anchors[anchor_indices, 2] - anchors[anchor_indices, 0] + 1e-7
    anc_ctr_x = anchors[anchor_indices, 1] + 0.5 * anc_width
    anc_ctr_y = anchors[anchor_indices, 0] + 0.5 * anc_height
    #
    gt_width = gt_boxes[gt_indices, 3] - gt_boxes[gt_indices, 1]
    gt_height = gt_boxes[gt_indices, 2] - gt_boxes[gt_indices, 0]
    gt_ctr_x = gt_boxes[gt_indices, 1] + 0.5 * gt_width
    gt_ctr_y = gt_boxes[gt_indices, 0] + 0.5 * gt_height
    #
    delta_x = (gt_ctr_x - anc_ctr_x) / anc_width
    delta_y = (gt_ctr_y - anc_ctr_y) / anc_height
    delta_w = np.log(gt_width / anc_width)
    delta_h = np.log(gt_height / anc_height)
    #
    bbox_deltas[anchor_indices, 0] = delta_y
    bbox_deltas[anchor_indices, 1] = delta_x
    bbox_deltas[anchor_indices, 2] = delta_h
    bbox_deltas[anchor_indices, 3] = delta_w
    #
    return bbox_deltas

def non_max_suppression(pred_bboxes, pred_labels, top_n_boxes=300):
    selected_indices = tf.image.non_max_suppression(pred_bboxes, pred_labels, top_n_boxes)
    selected_boxes = tf.gather(pred_bboxes, selected_indices)
    return selected_boxes.numpy()

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

def update_image_boundaries_with_padding(img_boundaries, padding):
    img_boundaries["top"] = padding["top"]
    img_boundaries["left"] = padding["left"]
    img_boundaries["bottom"] += padding["top"]
    img_boundaries["right"] += padding["left"]
    return img_boundaries

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

# image param => pillow image
def add_padding(image, top, right, bottom, left):
    width, height = image.size
    new_width = width + left + right
    new_height = height + top + bottom
    result = Image.new(image.mode, (new_width, new_height), (0, 0, 0))
    result.paste(image, (left, top))
    return result

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
