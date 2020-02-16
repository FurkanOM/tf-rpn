import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import helpers
import rpn

args = helpers.handle_args()
if args.handle_gpu:
    helpers.handle_gpu_compatibility()

apply_padding = True
hyper_params = {
    "anchor_ratios": [0.5, 1, 2],
    "anchor_scales": [16, 32, 64, 128, 256],
    "stride": 32,
    "nms_topn": 10,
    "total_pos_bboxes": 64,
}
hyper_params["anchor_count"] = len(hyper_params["anchor_ratios"]) * len(hyper_params["anchor_scales"])

base_model = VGG16(include_top=False)
if hyper_params["stride"] == 16:
    base_model = Sequential(base_model.layers[:-1])

model_path = rpn.get_model_path(hyper_params["stride"])
rpn_model = rpn.get_model(base_model, hyper_params)
rpn_model.load_weights(model_path)

VOC_test_data, _, hyper_params["total_labels"] = helpers.get_VOC_data("test")
hyper_params["total_labels"] += 1

if apply_padding:
    # If you want to use different dataset and don't know max height and width values
    # You can use calculate_max_height_width method in helpers
    max_height, max_width = helpers.VOC["max_height"], helpers.VOC["max_width"]
    VOC_test_data = VOC_test_data.map(lambda x : helpers.handle_padding(x, max_height, max_width))

for image_data in VOC_test_data:
    img = image_data["image"]
    input_img, img_params, gt_boxes, gt_labels = helpers.preprocessing(image_data, hyper_params, preprocess_input)
    pred_bbox_deltas, pred_labels = rpn_model.predict_on_batch(input_img)
    anchors = rpn.generate_anchors(img_params, hyper_params)
    pred_bbox_deltas = tf.reshape(pred_bbox_deltas, (-1, 4))
    pred_labels = tf.reshape(pred_labels, (-1, ))
    pred_bboxes = helpers.get_bboxes_from_deltas(anchors, pred_bbox_deltas)
    selected_bboxes = helpers.non_max_suppression(pred_bboxes, pred_labels, hyper_params)
    helpers.draw_bboxes(img, selected_bboxes)
