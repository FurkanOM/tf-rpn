import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import Helpers
import rpn
import numpy as np

args = Helpers.handle_args()
if args.handle_gpu:
    Helpers.handle_gpu_compatibility()

anchor_ratios = [0.5, 1, 2]
anchor_scales = [16, 32, 64, 128, 256]
anchor_count = len(anchor_ratios) * len(anchor_scales)
stride = vgg16_stride = 32
# If you want to use different dataset and don't know max height and width values
# You can use calculate_max_height_width method in helpers
max_height, max_width = Helpers.VOC["max_height"], Helpers.VOC["max_width"]
apply_padding = True

test_data = Helpers.get_pascal_VOC_data("test", Helpers.VOC["classes"])

base_model = VGG16(include_top=False)
if stride == 16:
    base_model = Sequential(base_model.layers[:-1])

model_path = Helpers.get_model_path(stride)
rpn_model = rpn.get_model(base_model, anchor_count)
rpn_model.load_weights(model_path)

for image_data in test_data:
    img = Helpers.get_image(image_data["image_path"], as_array=True)
    img_boundaries = Helpers.get_image_boundaries(img)
    if apply_padding:
        img, padding = Helpers.get_padded_img(img, max_height, max_width)
        img_boundaries = Helpers.update_image_boundaries_with_padding(img_boundaries, padding)
    input_img = rpn.get_input_img(img, preprocess_input)
    pred_bbox_deltas, pred_labels = rpn_model.predict_on_batch(input_img)
    anchors = rpn.get_anchors(img, anchor_ratios, anchor_scales, stride)
    pred_bboxes, pred_labels = rpn.get_predicted_bboxes_and_labels(anchor_count, anchors, pred_bbox_deltas, pred_labels)
    sorted_label_indices = pred_labels.argsort()[::-1]
    Helpers.draw_anchors(img, pred_bboxes[sorted_label_indices[0:5]])
