import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import helpers
import rpn

args = helpers.handle_args()
if args.handle_gpu:
    helpers.handle_gpu_compatibility()

batch_size = 1
hyper_params = helpers.get_hyper_params(nms_topn=10)

VOC_test_data, VOC_info = helpers.get_VOC_data("test")
labels = helpers.get_labels(VOC_info)
# We add 1 class for background
hyper_params["total_labels"] = len(labels) + 1

# If you want to use different dataset and don't know max height and width values
# You can use calculate_max_height_width method in helpers
max_height, max_width = helpers.VOC["max_height"], helpers.VOC["max_width"]
VOC_test_data = VOC_test_data.map(lambda x : helpers.preprocessing(x, max_height, max_width))

padded_shapes, padding_values = helpers.get_padded_batch_params()
VOC_test_data = VOC_test_data.padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padding_values)

base_model = VGG16(include_top=False)
if hyper_params["stride"] == 16:
    base_model = Sequential(base_model.layers[:-1])

rpn_model = rpn.get_model(base_model, hyper_params)

rpn_model_path = rpn.get_model_path(hyper_params["stride"])
rpn_model.load_weights(rpn_model_path)

for image_data in VOC_test_data:
    img, gt_boxes, gt_labels = image_data
    input_img, anchors = rpn.get_step_data(image_data, hyper_params, preprocess_input, mode="inference")
    rpn_bbox_deltas, rpn_labels = rpn_model.predict_on_batch(input_img)
    #
    anchors_shape = tf.shape(anchors)
    batch_size, anchor_row_size = anchors_shape[0], anchors_shape[1]
    rpn_bbox_deltas = tf.reshape(rpn_bbox_deltas, (batch_size, anchor_row_size, 4))
    rpn_labels = tf.reshape(rpn_labels, (batch_size, anchor_row_size, 1))
    #
    rpn_bboxes = helpers.get_bboxes_from_deltas(anchors, rpn_bbox_deltas)
    rpn_bboxes = tf.reshape(rpn_bboxes, (batch_size, anchor_row_size, 1, 4))
    #
    nms_bboxes = helpers.non_max_suppression(rpn_bboxes, rpn_labels, hyper_params)
    img_float32 = tf.image.convert_image_dtype(img, dtype=tf.float32)
    helpers.draw_bboxes(img_float32, nms_bboxes)
