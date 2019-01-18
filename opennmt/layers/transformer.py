"""Define layers related to the Google's Transformer model."""

import tensorflow as tf

from opennmt.utils.misc import get_compat_name


def _lower_triangle_mask(sequence_length, maximum_length=None, dtype=tf.float32):
  batch_size = tf.shape(sequence_length)[0]
  if maximum_length is None:
    maximum_length = tf.reduce_max(sequence_length)
  mask = tf.ones([batch_size, maximum_length, maximum_length], dtype=dtype)
  mask = tf.linalg.band_part(mask, -1, 0)
  return mask

def future_mask(sequence_length, maximum_length=None, dtype=tf.float32):
  """Returns a mask of future positions.

  Args:
    sequence_length: The sequence length.
    maximum_length: Optional size of the returned time dimension. Otherwise
      it is the maximum of :obj:`sequence_length`.
    dtype: The type of the mask tensor.

  Returns:
    A ``tf.Tensor`` of type :obj:`dtype` and shape
    ``[batch_size, max_length, max_length]``.
  """
  sequence_mask = tf.sequence_mask(sequence_length, maxlen=maximum_length, dtype=dtype)
  mask = _lower_triangle_mask(sequence_length, maximum_length=maximum_length, dtype=dtype)
  mask *= tf.expand_dims(sequence_mask, axis=1)
  return mask

def cumulative_average_mask(sequence_length, maximum_length=None, dtype=tf.float32):
  """Builds the mask to compute the cumulative average as described in
  https://arxiv.org/abs/1805.00631.

  Args:
    sequence_length: The sequence length.
    maximum_length: Optional size of the returned time dimension. Otherwise
      it is the maximum of :obj:`sequence_length`.
    dtype: The type of the mask tensor.

  Returns:
    A ``tf.Tensor`` of type :obj:`dtype` and shape
    ``[batch_size, max_length, max_length]``.
  """
  sequence_mask = tf.sequence_mask(sequence_length, maxlen=maximum_length, dtype=dtype)
  mask = _lower_triangle_mask(sequence_length, maximum_length=maximum_length, dtype=dtype)
  mask *= tf.expand_dims(sequence_mask, axis=2)
  weight = tf.range(1, tf.cast(tf.shape(mask)[1] + 1, dtype), dtype=dtype)
  mask /= tf.expand_dims(weight, 1)
  return mask

def cumulative_average(inputs, mask_or_step, cache=None):
  """Computes the cumulative average as described in
  https://arxiv.org/abs/1805.00631.

  Args:
    inputs: The sequence to average. A tensor of shape :math:`[B, T, D]`.
    mask_or_step: If :obj:`cache` is set, this is assumed to be the current step
      of the dynamic decoding. Otherwise, it is the mask matrix used to compute
      the cumulative average.
    cache: A dictionnary containing the cumulative average of the previous step.

  Returns:
    The cumulative average, a tensor of the same shape and type as :obj:`inputs`.
  """
  if cache is not None:
    step = tf.cast(mask_or_step, inputs.dtype)
    aa = (inputs + step * cache["prev_g"]) / (step + 1.0)
    cache["prev_g"] = aa
    return aa
  else:
    mask = mask_or_step
    return tf.matmul(mask, inputs)

def split_heads(inputs, num_heads):
  """Splits a tensor in depth.

  Args:
    inputs: A ``tf.Tensor`` of shape :math:`[B, T, D]`.
    num_heads: The number of heads :math:`H`.

  Returns:
    A ``tf.Tensor`` of shape :math:`[B, H, T, D / H]`.
  """
  static_shape = inputs.get_shape().as_list()
  depth = static_shape[-1]
  outputs = tf.reshape(
      inputs, [tf.shape(inputs)[0], tf.shape(inputs)[1], num_heads, depth // num_heads])
  outputs = tf.transpose(outputs, perm=[0, 2, 1, 3])
  return outputs

def combine_heads(inputs):
  """Concatenates heads.

  Args:
    inputs: A ``tf.Tensor`` of shape :math:`[B, H, T, D]`.

  Returns:
    A ``tf.Tensor`` of shape :math:`[B, T, D * H]`.
  """
  static_shape = inputs.get_shape().as_list()
  depth = static_shape[-1]
  num_heads = static_shape[1]
  outputs = tf.transpose(inputs, perm=[0, 2, 1, 3])
  outputs = tf.reshape(outputs, [tf.shape(outputs)[0], tf.shape(outputs)[1], depth * num_heads])
  return outputs

def dot_product_attention(queries,
                          keys,
                          values,
                          mask=None,
                          training=True,
                          dropout=0.0):
  """Computes the dot product attention.

  Args:
    queries: The sequence of queries. A tensor of shape :math:`[B, T_1, ...]`.
    keys: The sequence use to calculate attention scores. A tensor of shape
      :math:`[B, T_2, ...]`.
    values: The sequence to attend. A tensor of shape :math:`[B, T_2, ...]`.
    mask: A ``tf.Tensor`` applied to the dot product.
    training: Run in training mode.
    dropout: The probability to drop units from the inputs.

  Returns:
    A tuple ``(context vector, attention vector)``.
  """
  # Dot product between queries and keys.
  dot = tf.matmul(queries, keys, transpose_b=True)

  if mask is not None:
    mask = tf.cast(mask, tf.float32)
    dot = tf.cast(tf.cast(dot, tf.float32) * mask + ((1.0 - mask) * tf.float32.min), dot.dtype)

  # Compute attention weights.
  attn = tf.cast(tf.nn.softmax(tf.cast(dot, tf.float32)), dot.dtype)
  if training:
    drop_attn = tf.nn.dropout(attn, rate=dropout)
  else:
    drop_attn = attn

  # Compute attention context.
  context = tf.matmul(drop_attn, values)

  return context, attn


def multi_head_attention(num_heads,
                         queries,
                         memory,
                         num_units=None,
                         mask=None,
                         cache=None,
                         training=True,
                         dropout=0.0,
                         return_attention=False):
  """Computes the multi-head attention as described in
  https://arxiv.org/abs/1706.03762.

  Args:
    num_heads: The number of attention heads.
    queries: The sequence of queries. A tensor of shape :math:`[B, T_1, ...]`.
    memory: The sequence to attend. A tensor of shape :math:`[B, T_2, ...]`.
      If ``None``, computes self-attention.
    num_units: The number of hidden units. If not set, it is set to the input
      dimension.
    mask: A ``tf.Tensor`` applied to the dot product.
    cache: A dictionary containing pre-projected keys and values.
    training: Run in training mode.
    dropout: The probability to drop units from the inputs.
    return_attention: Return the attention head probabilities in addition to the
      context.

  Returns:
    The concatenated attention context of each head and the attention
    probabilities of the first head (if :obj:`return_attention` is set).
  """
  num_units = num_units or queries.get_shape().as_list()[-1]
  layer = MultiHeadAttention(
      num_heads,
      num_units,
      self_attention=memory is None,
      return_attention=return_attention,
      dropout=dropout,
      name=get_compat_name())
  return layer(
      queries,
      memory=memory,
      mask=mask,
      cache=cache,
      training=training)

def feed_forward(x, inner_dim, training=True, dropout=0.0):
  """Implements the Transformer's "Feed Forward" layer.

  .. math::

      ffn(x) = max(0, x*W_1 + b_1)*W_2 + b_2

  Args:
    x: The input.
    inner_dim: The number of units of the inner linear transformation.
    training: Run in training mode.
    dropout: The probability to drop units from the inner transformation.

  Returns:
    The transformed input.
  """
  input_dim = x.get_shape().as_list()[-1]
  layer = FeedForwardNetwork(
      inner_dim,
      input_dim,
      dropout=dropout,
      name=get_compat_name())
  return layer(x, training=training)

def norm(inputs):
  """Layer normalizes :obj:`inputs`."""
  return LayerNorm(name=get_compat_name(name="LayerNorm"))(inputs)

def drop_and_add(inputs,
                 outputs,
                 training=True,
                 dropout=0.1):
  """Drops units in the outputs and adds the previous values.

  Args:
    inputs: The input of the previous layer.
    outputs: The output of the previous layer.
    training: Run in training mode.
    dropout: The probability to drop units in :obj:`outputs`.

  Returns:
    The residual and normalized output.
  """
  if training:
    outputs = tf.nn.dropout(outputs, rate=dropout)

  input_dim = inputs.get_shape().as_list()[-1]
  output_dim = outputs.get_shape().as_list()[-1]

  if input_dim == output_dim:
    outputs += inputs
  return outputs


class LayerNorm(tf.keras.layers.Layer):
  """Layer normalization."""

  def __init__(self, epsilon=1e-6, name=None):
    """Initializes this layer.

    Args:
      epsilon: The epsilon value to use.
      name: An optional name for this layer.
    """
    super(LayerNorm, self).__init__(name=name)
    self.epsilon = epsilon

  def build(self, input_shape):
    """Creates the variables."""
    depth = input_shape.as_list()[-1]
    self.bias = self.add_variable(
        "beta", [depth], initializer=tf.keras.initializers.Constant(0))
    self.scale = self.add_variable(
        "gamma", [depth], initializer=tf.keras.initializers.Constant(1))
    super(LayerNorm, self).build(input_shape)

  def call(self, x):
    """Normalizes :obj:`x`."""
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + self.epsilon)
    return norm_x * self.scale + self.bias


class FeedForwardNetwork(tf.keras.layers.Layer):
  """Implements the Transformer's "Feed Forward" layer.

  .. math::

      ffn(x) = max(0, x*W_1 + b_1)*W_2 + b_2
  """

  def __init__(self, inner_dim, output_dim, dropout=0.1, name=None):
    """Initializes this layer.

    Args:
      inner_dim: The number of units of the inner linear transformation.
      output_dim: The number of units of the ouput linear transformation.
      dropout: The probability to drop units from the inner transformation.
      name: An optional name for this layer.
    """
    super(FeedForwardNetwork, self).__init__(name=name)
    self.inner = tf.keras.layers.Conv1D(inner_dim, 1, activation=tf.nn.relu, name="conv1d")
    self.outer = tf.keras.layers.Conv1D(output_dim, 1, name="conv1d_1")
    self.dropout = dropout

  def call(self, inputs, training=True):
    """Runs the layer."""
    inner = self.inner(inputs)
    if training:
      inner = tf.nn.dropout(inner, rate=self.dropout)
    return self.outer(inner)


class MultiHeadAttention(tf.keras.layers.Layer):
  """Computes the multi-head attention as described in
  https://arxiv.org/abs/1706.03762.
  """

  def __init__(self,
               num_heads,
               num_units,
               self_attention=False,
               return_attention=False,
               dropout=0.1,
               name=None):
    """Initializes this layers.

    Args:
      num_heads: The number of attention heads.
      num_units: The number of hidden units.
      self_attention: Whether this is a self-attention layer or not.
      return_attention: If ``True``, also return the attention weights of the
        first head.
      dropout: The probability to drop units from the inputs.
      name: An optional name for this layer.
    """
    super(MultiHeadAttention, self).__init__(name=name)
    if num_units % num_heads != 0:
      raise ValueError("Multi head attention requires that num_units is a"
                       " multiple of %s" % num_heads)
    self.num_heads = num_heads
    self.num_units = num_units
    self.dropout = dropout
    self.return_attention = return_attention
    if self_attention:
      self.linear_layers = [
          tf.keras.layers.Conv1D(num_units * 3, 1, name="conv1d"),
          tf.keras.layers.Conv1D(num_units, 1, name="conv1d_1")]
    else:
      self.linear_layers = [
          tf.keras.layers.Conv1D(num_units, 1, name="conv1d"),
          tf.keras.layers.Conv1D(num_units * 2, 1, name="conv1d_1"),
          tf.keras.layers.Conv1D(num_units, 1, name="conv1d_2")]

  def call(self, inputs, memory=None, mask=None, cache=None, training=True):
    """Runs the layer.

    Args:
      inputs: The sequence of queries. A tensor of shape :math:`[B, T_1, ...]`.
      memory: The sequence to attend. A tensor of shape :math:`[B, T_2, ...]`.
        If ``None``, computes self-attention.
      mask: A ``tf.Tensor`` applied to the dot product.
      cache: A dictionary containing pre-projected keys and values.
      training: Run in training mode.

    Returns:
      The concatenated attention context of each head and the attention
      probabilities of the first head (if return_attention is ``True``).
    """
    queries = self.linear_layers[0](inputs)

    if memory is None:
      queries, keys, values = tf.split(queries, 3, axis=-1)
      keys = split_heads(keys, self.num_heads)
      values = split_heads(values, self.num_heads)
      if cache is not None:
        keys = tf.concat([cache["self_keys"], keys], axis=2)
        values = tf.concat([cache["self_values"], values], axis=2)
        cache["self_keys"] = keys
        cache["self_values"] = values
    else:
      def _project_memory():
        keys, values = tf.split(self.linear_layers[1](memory), 2, axis=-1)
        keys = split_heads(keys, self.num_heads)
        values = split_heads(values, self.num_heads)
        return keys, values

      if cache is not None:
        keys, values = tf.cond(
            tf.equal(tf.shape(cache["memory_keys"])[2], 0),
            true_fn=_project_memory,
            false_fn=lambda: (cache["memory_keys"], cache["memory_values"]),
            name=self.name)
        cache["memory_keys"] = keys
        cache["memory_values"] = values
      else:
        keys, values = _project_memory()

    queries = split_heads(queries, self.num_heads)
    queries *= (self.num_units // self.num_heads)**-0.5
    if mask is not None:
      mask = tf.expand_dims(mask, 1)  # Broadcast on heads dimension.

    heads, attn = dot_product_attention(
        queries,
        keys,
        values,
        mask=mask,
        training=training,
        dropout=self.dropout)

    # Concatenate all heads output.
    combined = combine_heads(heads)
    outputs = self.linear_layers[-1](combined)
    if self.return_attention:
      return outputs, attn
    return outputs
