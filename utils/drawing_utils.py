"""Visualization helpers for debugging anchors and proposals."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import tensorflow as tf


def draw_grid_map(img: Any, grid_map: Any, stride: int) -> None:
    """Overlay the feature-map grid centers on top of an image.

    Args:
        img (Any): Image-like array accepted by `PIL.Image.fromarray`.
        grid_map (Any): Iterable grid coordinates describing anchor centers.
        stride (int): Pixel stride between neighboring grid locations.

    Returns:
        None: The function renders the visualization with Matplotlib.
    """
    image = Image.fromarray(img)
    draw = ImageDraw.Draw(image)
    for grid in grid_map:
        draw.rectangle((
            grid[0] + stride // 2 - 2,
            grid[1] + stride // 2 - 2,
            grid[2] + stride // 2 + 2,
            grid[3] + stride // 2 + 2), fill=(255, 255, 255, 0))
    plt.figure()
    plt.imshow(image)
    plt.show()


def draw_bboxes(imgs: tf.Tensor, bboxes: tf.Tensor) -> None:
    """Render bounding boxes on a batch of images.

    Args:
        imgs (tf.Tensor): Image batch with shape `(batch_size, height, width, 3)`.
        bboxes (tf.Tensor): Bounding boxes with shape `(batch_size, total_boxes, 4)`
            in normalized coordinates.

    Returns:
        None: Each image is rendered with Matplotlib.
    """
    colors = tf.constant([[1, 0, 0, 1]], dtype=tf.float32)
    imgs_with_bb = tf.image.draw_bounding_boxes(imgs, bboxes, colors)
    plt.figure()
    for img_with_bb in imgs_with_bb:
        plt.imshow(img_with_bb)
        plt.show()
