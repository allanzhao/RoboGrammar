import numpy as np
import tensorflow as tf
from gym.spaces import Discrete, Box, MultiDiscrete

def observation_placeholder(ob_space, batch_size=None, name='Ob'):
    '''
    Create placeholder to feed observations into of the size appropriate to the observation space

    Parameters:
    ----------

    ob_space: gym.Space     observation space

    batch_size: int         size of the batch to be fed into input. Can be left None in most cases.

    name: str               name of the placeholder

    Returns:
    -------

    tensorflow placeholder tensor
    '''

    assert isinstance(ob_space, Discrete) or isinstance(ob_space, Box) or isinstance(ob_space, MultiDiscrete), \
        'Can only deal with Discrete and Box observation spaces for now'

    dtype = ob_space.dtype
    if dtype == np.int8:
        dtype = np.uint8

    return tf.placeholder(shape=(batch_size,) + ob_space.shape, dtype=dtype, name=name)


def observation_input(ob_space, batch_size=None, name='Ob'):
    '''
    Create placeholder to feed observations into of the size appropriate to the observation space, and add input
    encoder of the appropriate type.
    '''

    placeholder = observation_placeholder(ob_space, batch_size, name)
    return placeholder, encode_observation(ob_space, placeholder)

def encode_observation(ob_space, placeholder):
    '''
    Encode input in the way that is appropriate to the observation space

    Parameters:
    ----------

    ob_space: gym.Space             observation space

    placeholder: tf.placeholder     observation input placeholder
    '''
    if isinstance(ob_space, Discrete):
        return tf.to_float(tf.one_hot(placeholder, ob_space.n))
    elif isinstance(ob_space, Box):
        return tf.to_float(placeholder)
    elif isinstance(ob_space, MultiDiscrete):
        placeholder = tf.cast(placeholder, tf.int32)
        one_hots = [tf.to_float(tf.one_hot(placeholder[..., i], ob_space.nvec[i])) for i in range(placeholder.shape[-1])]
        return tf.concat(one_hots, axis=-1)
    else:
        raise NotImplementedError

