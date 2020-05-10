import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from utils import io_utils, data_utils, train_utils, bbox_utils

args = io_utils.handle_args()
if args.handle_gpu:
    io_utils.handle_gpu_compatibility()

batch_size = 8
epochs = 50
load_weights = False
backbone = args.backbone
io_utils.is_valid_backbone(backbone)

if backbone == "vgg16":
    from models.rpn_vgg16 import get_model

hyper_params = train_utils.get_hyper_params(backbone)

train_data, dataset_info = data_utils.get_dataset("voc/2007", "train+validation")
val_data, _ = data_utils.get_dataset("voc/2007", "test")
train_total_items = data_utils.get_total_item_size(dataset_info, "train+validation")
val_total_items = data_utils.get_total_item_size(dataset_info, "test")
labels = data_utils.get_labels(dataset_info)
# We add 1 class for background
hyper_params["total_labels"] = len(labels) + 1
#
img_size = hyper_params["img_size"]
train_data = train_data.map(lambda x : data_utils.preprocessing(x, img_size, img_size, apply_augmentation=True))
val_data = val_data.map(lambda x : data_utils.preprocessing(x, img_size, img_size))

padded_shapes, padding_values = data_utils.get_padded_batch_params()
train_data = train_data.padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
val_data = val_data.padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padding_values)

anchors = bbox_utils.generate_anchors(hyper_params)
rpn_train_feed = train_utils.rpn_generator(train_data, anchors, hyper_params)
rpn_val_feed = train_utils.rpn_generator(val_data, anchors, hyper_params)

rpn_model, _ = get_model(hyper_params)
rpn_model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-5),
                  loss=[train_utils.reg_loss, train_utils.cls_loss])
# Load weights
model_path = io_utils.get_model_path()

if load_weights:
    rpn_model.load_weights(model_path)

checkpoint_callback = ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, save_weights_only=True)

step_size_train = train_utils.get_step_size(train_total_items, batch_size)
step_size_val = train_utils.get_step_size(val_total_items, batch_size)
rpn_model.fit(rpn_train_feed,
              steps_per_epoch=step_size_train,
              validation_data=rpn_val_feed,
              validation_steps=step_size_val,
              epochs=epochs,
              callbacks=[checkpoint_callback])
