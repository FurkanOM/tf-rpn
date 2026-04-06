"""Run Region Proposal Network inference on Pascal VOC data."""

from __future__ import annotations

import tensorflow as tf

from utils import bbox_utils, data_utils, drawing_utils, io_utils, train_utils


def main() -> None:
    """Run RPN inference from the command line.

    Returns:
        None: Predictions are rendered to the screen.
    """
    args = io_utils.handle_args()
    if args.handle_gpu:
        io_utils.handle_gpu_compatibility()

    batch_size = 4
    use_custom_images = False
    custom_image_path = "data/images/"
    backbone = args.backbone
    get_model = io_utils.get_rpn_model_builder(backbone)

    hyper_params = train_utils.get_hyper_params(backbone)

    test_data, dataset_info = data_utils.get_dataset("voc/2007", "test")
    labels = ["bg"] + list(data_utils.get_labels(dataset_info))
    hyper_params["total_labels"] = len(labels)
    img_size = hyper_params["img_size"]

    if use_custom_images:
        img_paths = data_utils.get_custom_imgs(custom_image_path)
        test_data = data_utils.build_custom_dataset(img_paths, img_size, img_size)
    else:
        test_data = data_utils.build_dataset(test_data, img_size, img_size, batch_size)

    test_data = test_data.padded_batch(
        batch_size,
        padded_shapes=data_utils.get_data_shapes(),
        padding_values=data_utils.get_padding_values()
    ) if use_custom_images else test_data

    rpn_model, _ = get_model(hyper_params)

    rpn_model_path = io_utils.get_model_path("rpn", backbone)
    rpn_model.load_weights(rpn_model_path, by_name=True)

    anchors = bbox_utils.generate_anchors(hyper_params)

    for image_data in test_data:
        imgs, _, _ = image_data
        rpn_bbox_deltas, rpn_labels = rpn_model.predict_on_batch(imgs)
        rpn_bbox_deltas = tf.reshape(rpn_bbox_deltas, (batch_size, -1, 4))
        rpn_labels = tf.reshape(rpn_labels, (batch_size, -1))
        rpn_bbox_deltas *= hyper_params["variances"]
        rpn_bboxes = bbox_utils.get_bboxes_from_deltas(anchors, rpn_bbox_deltas)
        _, top_indices = tf.nn.top_k(rpn_labels, 10)
        selected_rpn_bboxes = tf.gather(rpn_bboxes, top_indices, batch_dims=1)
        drawing_utils.draw_bboxes(imgs, selected_rpn_bboxes)


if __name__ == "__main__":
    main()
