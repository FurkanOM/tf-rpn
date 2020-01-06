import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model
import numpy as np
import Helpers

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

def generate_iou_map(anchors, gt_boxes, width, height):
    anchor_count = anchors.shape[0]
    gt_box_count = len(gt_boxes)
    iou_map = np.zeros((anchor_count, gt_box_count), dtype=np.float)
    for anc_index, anchor in enumerate(anchors):
        if anchor[0] < 0 or anchor [1] < 0 or anchor[2] > width or anchor[3] > height:
            continue
        for gt_index, gt_box_data in enumerate(gt_boxes):
            gt_box = gt_box_data["bbox"]
            iou = calculate_iou(anchor, gt_box)
            iou_map[anc_index, gt_index] = iou
    return iou_map

def get_bboxes_from_deltas(anchors, deltas):
    bboxes = np.zeros(anchors.shape)
    for index, delta in enumerate(deltas):
        anchor = anchors[index]
        delta_x, delta_y, delta_w, delta_h = delta
        #
        anc_width = anchor[2] - anchor[0]
        anc_height = anchor[3] - anchor[1]
        anc_ctr_x = anchor[0] + 0.5 * anc_width
        anc_ctr_y = anchor[1] + 0.5 * anc_height
        #
        bbox_width = np.exp(delta_x) * anc_width
        bbox_height = np.exp(delta_y) * anc_height
        bbox_ctr_x = (delta_x * anc_width) + anc_ctr_x
        bbox_ctr_y = (delta_y * anc_height) + anc_ctr_y
        #
        bbox_x1 = bbox_ctr_x - (0.5 * bbox_width)
        bbox_y1 = bbox_ctr_y - (0.5 * bbox_height)
        bbox_x2 = bbox_width + bbox_x1
        bbox_y2 = bbox_height + bbox_y1
        #
        bboxes[index] = [bbox_x1, bbox_y1, bbox_x2, bbox_y2]
    return bboxes

def get_deltas_from_bboxes(anchors, gt_boxes, pos_anchors):
    bbox_deltas = np.zeros(anchors.shape)
    for pos_anchor in pos_anchors:
        index, gt_box_index = pos_anchor
        anchor = anchors[index]
        gt_box_data = gt_boxes[gt_box_index]
        #
        anc_width = anchor[2] - anchor[0]
        anc_height = anchor[3] - anchor[1]
        anc_ctr_x = anchor[0] + 0.5 * anc_width
        anc_ctr_y = anchor[1] + 0.5 * anc_height
        #
        gt_box = gt_box_data["bbox"]
        gt_width = gt_box[2] - gt_box[0]
        gt_height = gt_box[3] - gt_box[1]
        gt_ctr_x = gt_box[0] + 0.5 * gt_width
        gt_ctr_y = gt_box[1] + 0.5 * gt_height
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

def get_image_params(img, stride):
    height, width, _ = img.shape
    output_height, output_width = height // stride, width // stride
    return height, width, output_height, output_width

def get_padded_img(img, max_height, max_width):
    height, width, _ = img.shape
    assert height <= max_height
    assert width <= max_width
    padding_height = max_height - height
    padding_width = max_width - width
    top = padding_height // 2
    bottom = padding_height - top
    left = padding_width // 2
    right = padding_width - left
    return np.pad(img, ((top, bottom), (left, right), (0,0)), mode='constant')

def preprocess_img(path, max_height=None, max_width=None, apply_padding=False):
    img = Helpers.get_image(path, as_array=True)
    if apply_padding:
        # Add padding to image depend of max values
        # This speed up processes and allow to batch predictions
        assert max_width != None
        assert max_height != None
        img = get_padded_img(img, max_height, max_width)
    return img

def postprocess_img(img, input_processor):
    processed_img = img.copy()
    processed_img = input_processor(processed_img)
    processed_img = np.expand_dims(processed_img, axis=0)
    return processed_img

def get_anchors(img, anchor_ratios, anchor_scales, stride):
    anchor_count = len(anchor_ratios) * len(anchor_scales)
    height, width, output_height, output_width = get_image_params(img, stride)
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
    return anchors

# This method was implemented by examining the python
# implementation of the code in the original article.
def get_bbox_deltas_and_labels(img, anchors, gt_boxes, anchor_count, stride):
    height, width, output_height, output_width = get_image_params(img, stride)
    #
    iou_map = generate_iou_map(anchors, gt_boxes, width, height)
    # any time => iou_map.reshape(output_height, output_width, anchor_count, len(gt_boxes))
    ################################################################
    max_indices_each_gt_box = iou_map.argmax(axis=1)
    # Positive and negative anchor numbers are 128 in original paper
    pos_anchor_number = 64
    # Set n pos anchor for every gt box
    use_max_n_indices_each_gt_box = max(pos_anchor_number // len(gt_boxes), 1)
    # You can use argsort(axis=1) for below operations
    # But in that case you need to check duplicated anchors for gt boxes
    pos_anchors = None
    for n_col in range(iou_map.shape[1]):
        # Get indices for gt box
        indices_for_column = np.where(max_indices_each_gt_box == n_col)[0]
        # Sort iou values descending order and get top n anchor indices for gt box
        sorted_indices_for_column = iou_map[indices_for_column][:,n_col].argsort()[::-1][:use_max_n_indices_each_gt_box]
        top_n_anchor_indices = indices_for_column[sorted_indices_for_column]
        # Init column indices aka gt box number for every anchor indices
        gt_box_indices = n_col + np.zeros(top_n_anchor_indices.shape, dtype=np.int32)
        # Handle the shape anchor_index, gt_box_index
        final_anchors = np.stack((top_n_anchor_indices, gt_box_indices)).transpose()
        # Place anchors to the pos anchors
        pos_anchors = final_anchors if pos_anchors is None else np.concatenate((pos_anchors, final_anchors), axis=0)
    #
    merged_iou_map = iou_map[np.arange(iou_map.shape[0]), max_indices_each_gt_box]
    neg_anchors = np.where(merged_iou_map < 0.3)[0]
    neg_anchors = neg_anchors[~np.isin(neg_anchors, pos_anchors[:,0])]
    #############################
    # Bbox calculation
    #############################
    bbox_deltas = get_deltas_from_bboxes(anchors, gt_boxes, pos_anchors)
    #############################
    # Label calculation
    #############################
    # labels => 1 object, 0 background, -1 neutral
    labels = -1 * np.ones((iou_map.shape[0], ), dtype=np.float32)
    labels[neg_anchors] = 0
    labels[pos_anchors[:,0]] = 1
    neg_anchors_count = len(neg_anchors)
    pos_anchors_count = len(pos_anchors[:,0])
    # We want to same number of positive and negative anchors
    # If there are more negative anchors than positive
    # Randomly change negative anchors to the neutral
    # until negative and positive anchors are equal
    if neg_anchors_count > pos_anchors_count:
        new_neutral_anchors = np.random.choice(neg_anchors, size=(neg_anchors_count - pos_anchors_count), replace=False)
        labels[new_neutral_anchors] = -1
    ############################################################
    bbox_deltas = bbox_deltas.reshape(output_height, output_width, anchor_count * 4)
    bbox_deltas = np.expand_dims(bbox_deltas, axis=0)
    labels = labels.reshape(output_height, output_width, anchor_count)
    labels = np.expand_dims(labels, axis=0)
    return bbox_deltas, labels

def generator(data,
              anchor_ratios,
              anchor_scales,
              stride,
              input_processor,
              max_height=None,
              max_width=None,
              apply_padding=False):
    while True:
        for index, image_data in enumerate(data):
            img = preprocess_img(image_data["image_path"], max_height, max_width, apply_padding)
            anchors = get_anchors(img, anchor_ratios, anchor_scales, stride)
            gt_boxes = image_data["gt_boxes"]
            anchor_count = len(anchor_ratios) * len(anchor_scales)
            bbox_deltas, labels = get_bbox_deltas_and_labels(img, anchors, gt_boxes, anchor_count, stride)
            input_img = postprocess_img(img, input_processor)
            yield input_img, [bbox_deltas, labels]

def get_model(base_model, anchor_count, learning_rate=0.001):
    output = Conv2D(512, (3, 3), activation="relu", padding="same", name="rpn_conv")(base_model.output)
    rpn_cls_output = Conv2D(anchor_count, (1, 1), activation="sigmoid", name="rpn_cls")(output)
    rpn_reg_output = Conv2D(anchor_count * 4, (1, 1), activation="linear", name="rpn_reg")(output)
    rpn_model = Model(inputs=base_model.input, outputs=[rpn_reg_output, rpn_cls_output])
    rpn_model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate, clipnorm=0.001),
                      loss=[rpn_reg_loss, rpn_cls_loss],
                      loss_weights=[10., 1.])
    return rpn_model
