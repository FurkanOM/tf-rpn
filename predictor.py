import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import Helpers
import rpn
import numpy as np

args = Helpers.handle_args()
if args.handle_gpu:
    Helpers.handle_gpu_compatibility()

epochs = 60
batch_size = 10
anchor_ratios = [0.5, 1, 2]
anchor_scales = [64, 128, 256]
anchor_count = len(anchor_ratios) * len(anchor_scales)
stride = vgg16_stride = 32

test_data = Helpers.get_pascal_VOC_data("test", Helpers.VOC["classes"])

base_model = VGG16(include_top=False)
rpn_model = rpn.get_model(base_model, anchor_count)

model_path = Helpers.get_model_path()
rpn_model.load_weights(model_path)

for image_data in test_data:
    img = rpn.preprocess_img(image_data["image_path"])
    img = rpn.postprocess_img(img, preprocess_input)
    pred_bbox_deltas, pred_labels = rpn_model.predict_on_batch(img)
