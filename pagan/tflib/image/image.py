import tensorflow as tf


def center_crop(image, size):
    # for image of shape [batch, height, width, channels] or [height, width, channels]
    if not isinstance(size, (tuple, list)):
        size = [size, size]
    offset_height = (tf.shape(image)[-3] - size[0]) // 2
    offset_width = (tf.shape(image)[-2] - size[1]) // 2
    return tf.image.crop_to_bounding_box(image, offset_height, offset_width, size[0], size[1])


