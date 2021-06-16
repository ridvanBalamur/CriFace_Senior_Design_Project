import numpy as np
import tensorflow as tf


def session(graph=None,
            allow_soft_placement=True,
            log_device_placement=False,
            allow_growth=True):
    """Return a Session with simple config."""
    config = tf.compat.v1.ConfigProto(allow_soft_placement=allow_soft_placement,
                            log_device_placement=log_device_placement)
    config.gpu_options.allow_growth = allow_growth
    return tf.compat.v1.Session(graph=graph, config=config)



