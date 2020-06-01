import tensorflow as tf
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def draw_grid_map(img, grid_map, stride):
    """Drawing grid intersection on given image.
    inputs:
        img = (height, width, channels)
        grid_map = (output_height * output_width, [y_index, x_index, y_index, x_index])
            tiled x, y coordinates
        stride = number of stride
    outputs:
        array = (height, width, channels)
    """
    image = Image.fromarray(img)
    draw = ImageDraw.Draw(image)
    counter = 0
    for grid in grid_map:
        draw.rectangle((
            grid[0] + stride // 2 - 2,
            grid[1] + stride // 2 - 2,
            grid[2] + stride // 2 + 2,
            grid[3] + stride // 2 + 2), fill=(255, 255, 255, 0))
        counter += 1
    plt.figure()
    plt.imshow(image)
    plt.show()

def draw_bboxes(imgs, bboxes):
    """Drawing bounding boxes on given images.
    inputs:
        imgs = (batch_size, height, width, channels)
        bboxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
            in normalized form [0, 1]
    """
    colors = tf.constant([[1, 0, 0, 1]], dtype=tf.float32)
    imgs_with_bb = tf.image.draw_bounding_boxes(imgs, bboxes, colors)
    plt.figure()
    for img_with_bb in imgs_with_bb:
        plt.imshow(img_with_bb)
        plt.show()
