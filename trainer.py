import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow_datasets as tfds
import Helpers
import rpn

args = Helpers.handle_args()
if args.handle_gpu:
    Helpers.handle_gpu_compatibility()

epochs = 100
batch_size = 2
anchor_ratios = [0.5, 1, 2]
anchor_scales = [16, 32, 64, 128, 256]
anchor_count = len(anchor_ratios) * len(anchor_scales)
stride = vgg16_stride = 32
# If you want to use different dataset and don't know max height and width values
# You can use calculate_max_height_width method in helpers
max_height, max_width = Helpers.VOC["max_height"], Helpers.VOC["max_width"]
apply_padding = True
load_weights = False

VOC_train = tfds.load("voc", split=tfds.Split.TRAIN)
VOC_val = tfds.load("voc", split=tfds.Split.VALIDATION)

rpn_train_feed = rpn.generator(VOC_train, anchor_ratios, anchor_scales, stride, preprocess_input, max_height=max_height, max_width=max_width, apply_padding=apply_padding)
rpn_val_feed = rpn.generator(VOC_val, anchor_ratios, anchor_scales, stride, preprocess_input, max_height=max_height, max_width=max_width, apply_padding=apply_padding)

base_model = VGG16(include_top=False, weights="imagenet")
if stride == 16:
    base_model = Sequential(base_model.layers[:-1])

model_path = Helpers.get_model_path(stride)
rpn_model = rpn.get_model(base_model, anchor_count)
if load_weights:
    rpn_model.load_weights(model_path)
rpn_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.000001),
                  loss=[rpn.rpn_reg_loss, rpn.rpn_cls_loss],
                  loss_weights=[10., 1.])

model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, save_weights_only=True, monitor="val_loss", mode="auto")
early_stopping = EarlyStopping(monitor="val_loss", patience=20, verbose=0, mode="auto")

step_size_train = len(train_data) // batch_size
step_size_val = len(val_data) // batch_size
rpn_model.fit_generator(generator=rpn_train_feed,
                        steps_per_epoch=step_size_train,
                        validation_data=rpn_val_feed,
                        validation_steps=step_size_val,
                        epochs=epochs,
                        callbacks=[early_stopping, model_checkpoint])
