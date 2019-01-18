"""Defines common layers."""

import tensorflow as tf


def sequence_mask(sequence_length, maximum_length=None, dtype=tf.float32):
  """Builds a sequence mask.

  Args:
    sequence_length: The sequence length.
    maximum_length: Optional size of the returned time dimension. Otherwise
      it is the maximum of :obj:`sequence_length`.
    dtype: The type of the mask tensor.

  Returns:
    A broadcastable ``tf.Tensor`` of type :obj:`dtype` and shape
    ``[batch_size, 1, max_length]``.
  """
  mask = tf.sequence_mask(sequence_length, maxlen=maximum_length, dtype=dtype)
  mask = tf.expand_dims(mask, 1)
  return mask
