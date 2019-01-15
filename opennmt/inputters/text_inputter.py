"""Define word-based embedders."""

import abc
import collections
import os
import six

import numpy as np
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector

from google.protobuf import text_format

from opennmt import constants, tokenizers
from opennmt.inputters.inputter import Inputter
from opennmt.utils.misc import count_lines
from opennmt.constants import PADDING_TOKEN


def visualize_embeddings(log_dir, embedding_var, vocabulary_file, num_oov_buckets=1):
  """Registers an embedding variable for visualization in TensorBoard.

  This function registers :obj:`embedding_var` in the ``projector_config.pbtxt``
  file and generates metadata from :obj:`vocabulary_file` to attach a label
  to each word ID.

  Args:
    log_dir: The active log directory.
    embedding_var: The embedding variable to visualize.
    vocabulary_file: The associated vocabulary file.
    num_oov_buckets: The number of additional unknown tokens.
  """
  # Copy vocabulary file to log_dir.
  basename = "%s.txt" % embedding_var.op.name.replace("/", "_")
  destination = os.path.join(log_dir, basename)
  tf.io.gfile.copy(vocabulary_file, destination, overwrite=True)

  # Append <unk> tokens.
  with tf.io.gfile.GFile(destination, mode="ab") as vocab:
    if num_oov_buckets == 1:
      vocab.write(b"<unk>\n")
    else:
      for i in range(num_oov_buckets):
        vocab.write(tf.compat.as_bytes("<unk%d>\n" % i))

  config = projector.ProjectorConfig()

  # If the projector file exists, load it.
  target = os.path.join(log_dir, "projector_config.pbtxt")
  if tf.io.gfile.exists(target):
    with tf.io.gfile.GFile(target, mode="rb") as target_file:
      text_format.Merge(target_file.read(), config)

  # If this embedding is already registered, just update the metadata path.
  exists = False
  for meta in config.embeddings:
    if meta.tensor_name == embedding_var.name:
      meta.metadata_path = basename
      exists = True
      break

  if not exists:
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = basename

  summary_writer = tf.summary.FileWriter(log_dir)

  projector.visualize_embeddings(summary_writer, config)

def load_pretrained_embeddings(embedding_file,
                               vocabulary_file,
                               num_oov_buckets=1,
                               with_header=True,
                               case_insensitive_embeddings=True):
  """Returns pretrained embeddings relative to the vocabulary.

  The :obj:`embedding_file` must have the following format:

  .. code-block:: text

      N M
      word1 val1 val2 ... valM
      word2 val1 val2 ... valM
      ...
      wordN val1 val2 ... valM

  or if :obj:`with_header` is ``False``:

  .. code-block:: text

      word1 val1 val2 ... valM
      word2 val1 val2 ... valM
      ...
      wordN val1 val2 ... valM

  This function will iterate on each embedding in :obj:`embedding_file` and
  assign the pretrained vector to the associated word in :obj:`vocabulary_file`
  if found. Otherwise, the embedding is ignored.

  If :obj:`case_insensitive_embeddings` is ``True``, word embeddings are assumed
  to be trained on lowercase data. In that case, word alignments are case
  insensitive meaning the pretrained word embedding for "the" will be assigned
  to "the", "The", "THE", or any other case variants included in
  :obj:`vocabulary_file`.

  Args:
    embedding_file: Path the embedding file. Entries will be matched against
      :obj:`vocabulary_file`.
    vocabulary_file: The vocabulary file containing one word per line.
    num_oov_buckets: The number of additional unknown tokens.
    with_header: ``True`` if the embedding file starts with a header line like
      in GloVe embedding files.
    case_insensitive_embeddings: ``True`` if embeddings are trained on lowercase
      data.

  Returns:
    A Numpy array of shape ``[vocabulary_size + num_oov_buckets, embedding_size]``.
  """
  # Map words to ids from the vocabulary.
  word_to_id = collections.defaultdict(list)
  with tf.io.gfile.GFile(vocabulary_file, mode="rb") as vocabulary:
    count = 0
    for word in vocabulary:
      word = word.strip()
      if case_insensitive_embeddings:
        word = word.lower()
      word_to_id[word].append(count)
      count += 1

  # Fill pretrained embedding matrix.
  with tf.io.gfile.GFile(embedding_file, mode="rb") as embedding:
    pretrained = None

    if with_header:
      next(embedding)

    for line in embedding:
      fields = line.strip().split()
      word = fields[0]

      if pretrained is None:
        pretrained = np.random.normal(
            size=(count + num_oov_buckets, len(fields) - 1))

      # Lookup word in the vocabulary.
      if word in word_to_id:
        ids = word_to_id[word]
        for index in ids:
          pretrained[index] = np.asarray(fields[1:])

  return pretrained

def tokens_to_chars(tokens, padding_value=None):
  """Splits tokens into unicode characters.

  Args:
    tokens: A string ``tf.Tensor`` of shape :math:`[..., T]`.
    padding_value: The value to use for padding. Defaults to the empty string.

  Returns:
    A string ``tf.Tensor`` of shape :math:`[..., T, W]`.
  """
  ragged = tf.strings.unicode_split(tokens, "UTF-8")
  return ragged.to_tensor(default_value=padding_value)

def _get_field(config, key, prefix=None, default=None, required=False):
  if prefix:
    key = "%s%s" % (prefix, key)
  value = config.get(key, default)
  if value is None and required:
    raise ValueError("Missing field '%s' in the data configuration" % key)
  return value


@six.add_metaclass(abc.ABCMeta)
class TextInputter(Inputter):
  """An abstract inputter that processes text."""

  def __init__(self, dtype=tf.float32):
    super(TextInputter, self).__init__(dtype=dtype)
    self.tokenizer = None

  def make_dataset(self, data_file):
    return tf.data.TextLineDataset(data_file)

  def get_dataset_size(self, data_file):
    return count_lines(data_file)

  def initialize(self, metadata, asset_dir=None, asset_prefix=""):
    tokenizer_config = _get_field(metadata, "tokenizer", prefix=asset_prefix)
    self.tokenizer = tokenizers.make_tokenizer(tokenizer_config)
    assets = {}
    if asset_dir:
      assets = self.tokenizer.make_assets(asset_dir, asset_prefix=asset_prefix)
    return assets

  def make_features(self, element=None, features=None):
    """Tokenizes raw text."""
    if features is None:
      features = {}
    if "tokens" in features:
      return features
    if element is None:
      raise ValueError("Missing element")
    tokens = self.tokenizer.tokenize(element)
    features["length"] = tf.shape(tokens)[0]
    features["tokens"] = tokens
    return features

  def _get_receiver_tensors(self):
    return {
        "tokens": tf.placeholder(tf.string, shape=(None, None)),
        "length": tf.placeholder(tf.int32, shape=(None,))
    }

  @abc.abstractmethod
  def make_inputs(self, features, training=True):
    raise NotImplementedError()


class WordEmbedder(TextInputter):
  """Simple word embedder."""

  def __init__(self, embedding_size=None, dropout=0.0, dtype=tf.float32):
    """Initializes the parameters of the word embedder.

    Args:
      embedding_size: The size of the resulting embedding.
        If ``None``, an embedding file must be provided.
      dropout: The probability to drop units in the embedding.
      dtype: The embedding type.
    """
    super(WordEmbedder, self).__init__(dtype=dtype)
    self.vocabulary_size = None
    self.embedding_size = embedding_size
    self.embedding_file = None
    self.trainable = True
    self.embeddings = None
    self.dropout = tf.keras.layers.Dropout(dropout)
    self.num_oov_buckets = 1

  def initialize(self, metadata, asset_dir=None, asset_prefix=""):
    assets = super(WordEmbedder, self).initialize(
        metadata, asset_dir=asset_dir, asset_prefix=asset_prefix)
    self.vocabulary_file = _get_field(metadata, "vocabulary", prefix=asset_prefix, required=True)
    self.vocabulary_size = count_lines(self.vocabulary_file) + self.num_oov_buckets
    self.vocabulary = self.vocabulary_lookup()

    embedding = _get_field(metadata, "embedding", prefix=asset_prefix)
    if embedding is None and self.embedding_size is None:
      raise ValueError("embedding_size must be set")
    if embedding is not None:
      self.embedding_file = embedding["path"]
      self.trainable = embedding.get("trainable", True)
      self.embedding_file_with_header = embedding.get("with_header", True)
      self.case_insensitive_embeddings = embedding.get("case_insensitive", True)

    return assets

  def _build(self):
    if self.embedding_file:
      pretrained = load_pretrained_embeddings(
          self.embedding_file,
          self.vocabulary_file,
          num_oov_buckets=self.num_oov_buckets,
          with_header=self.embedding_file_with_header,
          case_insensitive_embeddings=self.case_insensitive_embeddings)
      self.embedding_size = pretrained.shape[-1]
      initializer = tf.constant(pretrained.astype(self.dtype.as_numpy_dtype()))
    else:
      shape = [self.vocabulary_size, self.embedding_size]
      initializer = lambda: tf.keras.initializers.glorot_uniform()(shape, dtype=self.dtype)
    self.embeddings = tf.Variable(
        initial_value=initializer,
        trainable=self.trainable,
        name="w_embs",
        dtype=self.dtype)

  def vocabulary_lookup(self):
    """Returns a lookup table mapping string to index."""
    return tf.contrib.lookup.index_table_from_file(
        self.vocabulary_file,
        vocab_size=self.vocabulary_size - self.num_oov_buckets,
        num_oov_buckets=self.num_oov_buckets)

  def vocabulary_lookup_reverse(self):
    """Returns a lookup table mapping index to string."""
    return tf.contrib.lookup.index_to_string_table_from_file(
        self.vocabulary_file,
        vocab_size=self.vocabulary_size - self.num_oov_buckets,
        default_value=constants.UNKNOWN_TOKEN)

  def make_features(self, element=None, features=None):
    """Converts words tokens to ids."""
    features = super(WordEmbedder, self).make_features(element=element, features=features)
    if "ids" in features:
      return features
    ids = self.vocabulary.lookup(features["tokens"])
    if not self.is_target:
      features["ids"] = ids
    else:
      bos = tf.constant([constants.START_OF_SENTENCE_ID], dtype=ids.dtype)
      eos = tf.constant([constants.END_OF_SENTENCE_ID], dtype=ids.dtype)
      features["ids"] = tf.concat([bos, ids], axis=0)
      features["ids_out"] = tf.concat([ids, eos], axis=0)
      features["length"] += 1  # Increment length accordingly.
    return features

  def visualize(self, log_dir):
    visualize_embeddings(
        log_dir,
        self.embeddings,
        self.vocabulary_file,
        num_oov_buckets=self.num_oov_buckets)

  def make_inputs(self, features, training=True):
    outputs = tf.nn.embedding_lookup(self.embeddings, features["ids"])
    if training:
      outputs = self.dropout(outputs)
    return outputs


@six.add_metaclass(abc.ABCMeta)
class CharEmbedder(TextInputter):
  """Base class for character-aware inputters."""

  def __init__(self, embedding_size, dropout=0.0, dtype=tf.float32):
    """Initializes the parameters of the character embedder.

    Args:
      embedding_size: The size of the character embedding.
      dropout: The probability to drop units in the embedding.
      dtype: The embedding type.
    """
    super(CharEmbedder, self).__init__(dtype=dtype)
    self.embedding_size = embedding_size
    self.dropout = dropout
    self.num_oov_buckets = 1

  def initialize(self, metadata, asset_dir=None, asset_prefix=""):
    assets = super(CharEmbedder, self).initialize(
        metadata, asset_dir=asset_dir, asset_prefix=asset_prefix)
    self.vocabulary_file = _get_field(metadata, "vocabulary", prefix=asset_prefix, required=True)
    self.vocabulary_size = count_lines(self.vocabulary_file) + self.num_oov_buckets
    self.vocabulary = tf.contrib.lookup.index_table_from_file(
        self.vocabulary_file,
        vocab_size=self.vocabulary_size - self.num_oov_buckets,
        num_oov_buckets=self.num_oov_buckets)
    return assets

  def _build(self):
    shape = [self.vocabulary_size, self.embedding_size]
    initializer = lambda: tf.keras.initializers.glorot_uniform()(shape, dtype=self.dtype)
    self.embeddings = tf.Variable(
        initial_value=initializer,
        name="w_char_embs",
        dtype=self.dtype)

  def make_features(self, element=None, features=None):
    """Converts words to characters."""
    if features is None:
      features = {}
    if "char_ids" in features:
      return features
    if "chars" in features:
      chars = features["chars"]
    else:
      features = super(CharEmbedder, self).make_features(element=element, features=features)
      chars = tokens_to_chars(features["tokens"], padding_value=PADDING_TOKEN)
    features["char_ids"] = self.vocabulary.lookup(chars)
    return features

  def visualize(self, log_dir):
    visualize_embeddings(
        log_dir,
        self.embeddings,
        self.vocabulary_file,
        num_oov_buckets=self.num_oov_buckets)

  def _embed(self, inputs, training):
    mask = tf.math.not_equal(inputs, 0)
    outputs = tf.nn.embedding_lookup(self.embeddings, inputs)
    if training:
      outputs = tf.nn.dropout(outputs, rate=self.dropout)
    return outputs, mask

  @abc.abstractmethod
  def make_inputs(self, features, training=True):
    raise NotImplementedError()


class CharConvEmbedder(CharEmbedder):
  """An inputter that applies a convolution on characters embeddings."""

  def __init__(self,
               embedding_size,
               num_outputs,
               kernel_size=5,
               stride=3,
               dropout=0.0,
               dtype=tf.float32):
    """Initializes the parameters of the character convolution embedder.

    Args:
      embedding_size: The size of the character embedding.
      num_outputs: The dimension of the convolution output space.
      kernel_size: Length of the convolution window.
      stride: Length of the convolution stride.
      dropout: The probability to drop units in the embedding.
      dtype: The embedding type.
    """
    super(CharConvEmbedder, self).__init__(
        embedding_size, dropout=dropout, dtype=dtype)
    self.output_size = num_outputs
    self.conv = tf.keras.layers.Conv1D(
        num_outputs,
        kernel_size,
        strides=stride,
        padding="same",
        dtype=dtype)

  def _build(self):
    super(CharConvEmbedder, self)._build()
    self.conv.build(tf.TensorShape([None, None, self.embedding_size]))

  def make_inputs(self, features, training=True):
    inputs = features["char_ids"]
    inputs_shape = tf.shape(inputs)
    inputs, _ = self._embed(inputs, training)
    # Merge batch and sequence timesteps dimensions.
    inputs = tf.reshape(inputs, [-1, inputs_shape[2], self.embedding_size])
    outputs = self.conv(inputs)
    # Max pooling over depth.
    outputs = tf.reduce_max(outputs, axis=1)
    # Split batch and sequence timesteps dimensions.
    outputs = tf.reshape(outputs, [-1, inputs_shape[1], self.output_size])
    return outputs


class CharRNNEmbedder(CharEmbedder):
  """An inputter that runs a single RNN layer over character embeddings."""

  def __init__(self,
               embedding_size,
               num_units,
               dropout=0.2,
               cell_class=tf.keras.layers.LSTMCell,
               dtype=tf.float32):
    """Initializes the parameters of the character RNN embedder.

    Args:
      embedding_size: The size of the character embedding.
      num_units: The number of units in the RNN layer.
      dropout: The probability to drop units in the embedding and the RNN
        outputs.
      cell_class: The inner cell class or a callable taking :obj:`num_units` as
        argument and returning a cell.
      dtype: The embedding type.
    """
    super(CharRNNEmbedder, self).__init__(
        embedding_size,
        dropout=dropout,
        dtype=dtype)
    self.num_units = num_units
    self.rnn = tf.keras.layers.RNN(cell_class(num_units, dtype=dtype))

  def _build(self):
    super(CharRNNEmbedder, self)._build()
    self.rnn.build(tf.TensorShape([None, None, self.embedding_size]))

  def make_inputs(self, features, training=True):
    inputs = features["char_ids"]
    inputs_shape = tf.shape(inputs)
    inputs = tf.reshape(inputs, [-1, inputs_shape[2]])
    inputs, mask = self._embed(inputs, training)
    outputs = self.rnn(inputs, mask=mask, training=training)
    if training:
      outputs = tf.nn.dropout(outputs, rate=self.dropout)
    outputs = tf.reshape(outputs, [-1, inputs_shape[1], self.num_units])
    return outputs
