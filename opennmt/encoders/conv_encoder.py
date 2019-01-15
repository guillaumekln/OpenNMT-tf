"""Define convolution-based encoders."""

import tensorflow as tf

from opennmt.encoders.encoder import Encoder
from opennmt.layers.position import PositionEmbedder


class ConvEncoder(Encoder):
  """An encoder that applies a convolution over the input sequence
  as described in https://arxiv.org/abs/1611.02344.
  """

  def __init__(self,
               num_layers,
               num_units,
               kernel_size=3,
               dropout=0.3,
               position_encoder=None):
    """Initializes the parameters of the encoder.

    Args:
      num_layers: The number of convolutional layers.
      num_units: The number of output filters.
      kernel_size: The kernel size.
      dropout: The probability to drop units from the inputs.
      position_encoder: The :class:`opennmt.layers.position.PositionEncoder` to
        apply on inputs. If ``None``, defaults to
        :class:`opennmt.layers.position.PositionEmbedder`.
    """
    super(ConvEncoder, self).__init__()
    self.dropout = dropout
    self.position_encoder = position_encoder
    if self.position_encoder is None:
      self.position_encoder = PositionEmbedder()
    self.conv_a = [
        tf.keras.layers.Conv1D(num_units, kernel_size, padding="same")
        for _ in range(num_layers)]
    self.conv_c = [
        tf.keras.layers.Conv1D(num_units, kernel_size, padding="same")
        for _ in range(num_layers)]

  def call(self, inputs, sequence_length=None, training=True):
    inputs = self.position_encoder(inputs)
    if training:
      inputs = tf.nn.dropout(inputs, rate=self.dropout)

    with tf.name_scope("cnn_a/"):
      cnn_a = _cnn_stack(inputs, self.conv_a)
    with tf.name_scope("cnn_c/"):
      cnn_c = _cnn_stack(inputs, self.conv_c)

    encoder_output = cnn_a
    encoder_state = tf.reduce_mean(cnn_c, axis=1)
    return (encoder_output, encoder_state, sequence_length)


def _cnn_stack(inputs, layers):
  next_input = inputs
  for i, layer in enumerate(layers):
    outputs = layer(next_input)
    # Add residual connections past the first layer.
    if i > 0:
      outputs += next_input
    next_input = tf.tanh(outputs)
  return next_input
