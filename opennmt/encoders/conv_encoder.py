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
    self.num_layers = num_layers
    self.num_units = num_units
    self.kernel_size = kernel_size
    self.dropout = dropout
    self.position_encoder = position_encoder
    if self.position_encoder is None:
      self.position_encoder = PositionEmbedder()

  def call(self, inputs, sequence_length=None, training=True):
    inputs = self.position_encoder(inputs)
    if training:
      inputs = tf.nn.dropout(inputs, rate=self.dropout)

    with tf.variable_scope("cnn_a"):
      cnn_a = self._cnn_stack(inputs)
    with tf.variable_scope("cnn_c"):
      cnn_c = self._cnn_stack(inputs)

    encoder_output = cnn_a
    encoder_state = tf.reduce_mean(cnn_c, axis=1)

    return (encoder_output, encoder_state, sequence_length)

  def _cnn_stack(self, inputs):
    next_input = inputs

    for l in range(self.num_layers):
      outputs = tf.layers.conv1d(
          next_input,
          self.num_units,
          self.kernel_size,
          padding="same")

      # Add residual connections past the first layer.
      if l > 0:
        outputs += next_input

      next_input = tf.tanh(outputs)

    return next_input
