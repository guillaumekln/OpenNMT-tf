"""Define the self-attention encoder."""

import tensorflow as tf

from opennmt.encoders.encoder import Encoder
from opennmt.layers import common, transformer
from opennmt.layers.position import SinusoidalPositionEncoder


class SelfAttentionEncoder(Encoder):
  """Encoder using self-attention as described in
  https://arxiv.org/abs/1706.03762.
  """

  def __init__(self,
               num_layers,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               relu_dropout=0.1,
               position_encoder=None):
    """Initializes the parameters of the encoder.

    Args:
      num_layers: The number of layers.
      num_units: The number of hidden units.
      num_heads: The number of heads in the multi-head attention.
      ffn_inner_dim: The number of units of the inner linear transformation
        in the feed forward layer.
      dropout: The probability to drop units from the outputs.
      attention_dropout: The probability to drop units from the attention.
      relu_dropout: The probability to drop units from the ReLU activation in
        the feed forward layer.
      position_encoder: The :class:`opennmt.layers.position.PositionEncoder` to
        apply on inputs. If ``None``, defaults to
        :class:`opennmt.layers.position.SinusoidalPositionEncoder`.
    """
    super(SelfAttentionEncoder, self).__init__()
    self.num_layers = num_layers
    self.num_units = num_units
    self.num_heads = num_heads
    self.ffn_inner_dim = ffn_inner_dim
    self.dropout = dropout
    self.attention_dropout = attention_dropout
    self.relu_dropout = relu_dropout
    self.position_encoder = position_encoder
    if self.position_encoder is None:
      self.position_encoder = SinusoidalPositionEncoder()

  def call(self, inputs, sequence_length=None, training=True):
    inputs *= self.num_units**0.5
    inputs = self.position_encoder(inputs)
    if training:
      inputs = tf.nn.dropout(inputs, rate=self.dropout)

    mask = common.sequence_mask(
        sequence_length, maximum_length=tf.shape(inputs)[1], dtype=tf.float32)

    state = ()

    for l in range(self.num_layers):
      with tf.variable_scope("layer_{}".format(l)):
        with tf.variable_scope("multi_head"):
          context = transformer.multi_head_attention(
              self.num_heads,
              transformer.norm(inputs),
              None,
              num_units=self.num_units,
              mask=mask,
              training=training,
              dropout=self.attention_dropout)
          context = transformer.drop_and_add(
              inputs,
              context,
              training=training,
              dropout=self.dropout)

        with tf.variable_scope("ffn"):
          transformed = transformer.feed_forward(
              transformer.norm(context),
              self.ffn_inner_dim,
              training=training,
              dropout=self.relu_dropout)
          transformed = transformer.drop_and_add(
              context,
              transformed,
              training=training,
              dropout=self.dropout)

        inputs = transformed
        state += (tf.reduce_mean(inputs, axis=1),)

    outputs = transformer.norm(inputs)
    return (outputs, state, sequence_length)
