import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow_datasets as tfds
import Helpers
import rpn

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

VOC_test = tfds.load("voc", split=tfds.Split.TEST)

base_model = VGG16(include_top=False)
if stride == 16:
    base_model = Sequential(base_model.layers[:-1])

model_path = Helpers.get_model_path(stride)
rpn_model = rpn.get_model(base_model, anchor_count)
rpn_model.load_weights(model_path)

for image_data in VOC_test:
    img = image_data["image"].numpy()
    img_boundaries = Helpers.get_image_boundaries(img)
    if apply_padding:
        img, padding = Helpers.get_padded_img(img, max_height, max_width)
        img_boundaries = Helpers.update_image_boundaries_with_padding(img_boundaries, padding)
    input_img = rpn.get_input_img(img, preprocess_input)
    pred_bbox_deltas, pred_labels = rpn_model.predict_on_batch(input_img)
    anchors = rpn.get_anchors(img, anchor_ratios, anchor_scales, stride)
    pred_bboxes, pred_labels = rpn.get_predicted_bboxes_and_labels(anchor_count, anchors, pred_bbox_deltas, pred_labels)
    selected_bboxes, selected_labels = rpn.non_max_suppression(pred_bboxes, pred_labels, top_n_boxes=10)
    Helpers.draw_anchors(img, selected_bboxes)
