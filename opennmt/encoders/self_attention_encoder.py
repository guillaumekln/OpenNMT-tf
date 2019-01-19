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
    self.num_units = num_units
    self.dropout = dropout
    self.position_encoder = position_encoder
    if self.position_encoder is None:
      self.position_encoder = SinusoidalPositionEncoder()
    self.layer_norm = transformer.LayerNorm(name="LayerNorm")
    self.layers = [
        _SelfAttentionEncoderLayer(
            num_heads,
            num_units,
            ffn_inner_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            relu_dropout=relu_dropout,
            name="layer_%d" % i)
        for i in range(num_layers)]

  def call(self, inputs, sequence_length=None, training=True):
    inputs *= self.num_units**0.5
    inputs = self.position_encoder(inputs)
    if training:
      inputs = tf.nn.dropout(inputs, rate=self.dropout)
    mask = common.sequence_mask(
        sequence_length, maximum_length=tf.shape(inputs)[1], dtype=tf.float32)

    state = []
    for layer in self.layers:
      inputs = layer(inputs, mask=mask, training=training)
      state.append(tf.reduce_mean(inputs, axis=1))

    outputs = self.layer_norm(inputs)
    return (outputs, tuple(state), sequence_length)


class _SelfAttentionEncoderLayer(tf.keras.layers.Layer):
  """Implements one self-attention encoding layer."""

  def __init__(self,
               num_heads,
               num_units,
               ffn_inner_dim,
               dropout=0.1,
               attention_dropout=0.1,
               relu_dropout=0.1,
               **kwargs):
    super(_SelfAttentionEncoderLayer, self).__init__(**kwargs)
    self.dropout = dropout
    self.self_attention = transformer.MultiHeadAttention(
        num_heads,
        num_units,
        self_attention=True,
        normalize_input=True,
        dropout=attention_dropout,
        name="multi_head")
    self.ffn = transformer.FeedForwardNetwork(
        ffn_inner_dim,
        num_units,
        normalize_input=True,
        dropout=relu_dropout,
        name="ffn")

  def call(self, x, mask=None, training=True):
    """Runs the encoder layer."""
    y = self.self_attention(x, mask=mask, training=training)
    x = transformer.drop_and_add(x, y, training=training, dropout=self.dropout)
    y = self.ffn(x, training=training)
    x = transformer.drop_and_add(x, y, training=training, dropout=self.dropout)
    return x
