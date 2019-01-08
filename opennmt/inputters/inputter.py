"""Define generic inputters."""

import abc
import six

import tensorflow as tf

from opennmt.layers.reducer import ConcatReducer, JoinReducer


@six.add_metaclass(abc.ABCMeta)
class Inputter(object):
  """Base class for inputters."""

  def __init__(self, dtype=tf.float32):
    self.dtype = dtype
    self._is_target = False

  @property
  def num_outputs(self):
    """How many parallel outputs does this inputter produce."""
    return 1

  @property
  def is_target(self):
    """Returns ``True`` if this inputter is loading target inputs."""
    return self._is_target

  @is_target.setter
  def is_target(self, value):
    self._is_target = value

  def get_length(self, unused_features):
    """Returns the length of the input features, if defined."""
    return None

  @abc.abstractmethod
  def make_dataset(self, data_file):
    """Creates the dataset required by this inputter.

    Args:
      data_file: The data file.

    Returns:
      A ``tf.data.Dataset``.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def get_dataset_size(self, data_file):
    """Returns the size of the dataset.

    Args:
      data_file: The data file.

    Returns:
      The total size.
    """
    raise NotImplementedError()

  def get_serving_input_receiver(self):
    """Returns a serving input receiver for this inputter.

    Returns:
      A ``tf.estimator.export.ServingInputReceiver``.
    """
    if self._is_target:
      raise ValueError("Target inputters do not define a serving input")
    receiver_tensors = self._get_receiver_tensors()
    features = self.make_features(features=receiver_tensors.copy())
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

  @abc.abstractmethod
  def _get_receiver_tensors(self):
    """Returns the input receiver for serving."""
    raise NotImplementedError()

  def initialize(self, metadata, asset_dir=None, asset_prefix=""):
    """Initializes the inputter within the current graph.

    For example, one can create lookup tables in this method
    for their initializer to be added to the current graph
    ``TABLE_INITIALIZERS`` collection.

    Args:
      metadata: A dictionary containing additional metadata set
        by the user.
      asset_dir: The directory where assets can be written. If ``None``, no
        assets are returned.
      asset_prefix: The prefix to attach to assets filename.

    Returns:
      A dictionary containing additional assets used by the inputter.
    """
    _ = metadata
    _ = asset_dir
    _ = asset_prefix
    return {}

  @abc.abstractmethod
  def make_features(self, element=None, features=None):
    """Creates features from data.

    Args:
      element: An element from the dataset.
      features: An optional dictionary of features to augment.

    Returns:
      A dictionary of ``tf.Tensor``.
    """
    raise NotImplementedError()

  def visualize(self, log_dir):
    """Visualizes the transformation, usually embeddings.

    Args:
      log_dir: The active log directory.
    """
    pass

  @abc.abstractmethod
  def __call__(self, features, training=True):
    """Creates the model input from the features.

    Args:
      features: A dictionary of ``tf.Tensor``.
      training: Training mode.

    Returns:
      The model input.
    """
    raise NotImplementedError()


@six.add_metaclass(abc.ABCMeta)
class MultiInputter(Inputter):
  """An inputter that gathers multiple inputters."""

  def __init__(self, inputters, reducer=None):
    if not isinstance(inputters, list) or not inputters:
      raise ValueError("inputters must be a non empty list")
    dtype = inputters[0].dtype
    for inputter in inputters:
      if inputter.dtype != dtype:
        raise TypeError("All inputters must have the same dtype")
    super(MultiInputter, self).__init__(dtype=dtype)
    self.inputters = inputters
    self.reducer = reducer

  @property
  def num_outputs(self):
    if self.reducer is None or isinstance(self.reducer, JoinReducer):
      return len(self.inputters)
    return 1

  @abc.abstractmethod
  def make_dataset(self, data_file):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_dataset_size(self, data_file):
    raise NotImplementedError()

  def initialize(self, metadata, asset_dir=None, asset_prefix=""):
    assets = {}
    for i, inputter in enumerate(self.inputters):
      assets.update(inputter.initialize(
          metadata, asset_dir=asset_dir, asset_prefix="%s%d_" % (asset_prefix, i + 1)))
    return assets

  def visualize(self, log_dir):
    for i, inputter in enumerate(self.inputters):
      with tf.variable_scope("inputter_{}".format(i)):
        inputter.visualize(log_dir)

  @abc.abstractmethod
  def _get_receiver_tensors(self):
    raise NotImplementedError()


class ParallelInputter(MultiInputter):
  """An multi inputter that process parallel data."""

  def __init__(self, inputters, reducer=None):
    """Initializes a parallel inputter.

    Args:
      inputters: A list of :class:`opennmt.inputters.inputter.Inputter`.
      reducer: A :class:`opennmt.layers.reducer.Reducer` to merge all inputs. If
        set, parallel inputs are assumed to have the same length.
    """
    super(ParallelInputter, self).__init__(inputters, reducer=reducer)

  def get_length(self, features):
    lengths = []
    for i, inputter in enumerate(self.inputters):
      sub_features = _extract_suffixed_keys(features, "_%d" % i)
      lengths.append(inputter.get_length(sub_features))
    if self.reducer is None:
      return lengths
    else:
      return lengths[0]

  def make_dataset(self, data_file):
    if not isinstance(data_file, list) or len(data_file) != len(self.inputters):
      raise ValueError("The number of data files must be the same as the number of inputters")
    datasets = [
        inputter.make_dataset(data)
        for inputter, data in zip(self.inputters, data_file)]
    return tf.data.Dataset.zip(tuple(datasets))

  def get_dataset_size(self, data_file):
    if not isinstance(data_file, list) or len(data_file) != len(self.inputters):
      raise ValueError("The number of data files must be the same as the number of inputters")
    dataset_sizes = [
        inputter.get_dataset_size(data)
        for inputter, data in zip(self.inputters, data_file)]
    dataset_size = dataset_sizes[0]
    for size in dataset_sizes:
      if size != dataset_size:
        raise RuntimeError("The parallel data files do not have the same size")
    return dataset_size

  def _get_receiver_tensors(self):
    receiver_tensors = {}
    for i, inputter in enumerate(self.inputters):
      tensors = inputter._get_receiver_tensors()  # pylint: disable=protected-access
      for key, value in six.iteritems(tensors):
        receiver_tensors["{}_{}".format(key, i)] = value
    return receiver_tensors

  def make_features(self, element=None, features=None):
    if features is None:
      features = {}
    all_features = {}
    for i, inputter in enumerate(self.inputters):
      suffix = "_%d" % i
      sub_features = inputter.make_features(
          element=element[i] if element is not None else None,
          features=_extract_suffixed_keys(features, suffix))
      for key, value in six.iteritems(sub_features):
        all_features["%s%s" % (key, suffix)] = value
    return all_features

  def __call__(self, features, training=True):
    inputs = []
    for i, inputter in enumerate(self.inputters):
      with tf.variable_scope("inputter_{}".format(i)):
        sub_features = _extract_suffixed_keys(features, "_%d" % i)
        inputs.append(inputter(sub_features, training=training))
    if self.reducer is not None:
      inputs = self.reducer(inputs)
    return inputs


class MixedInputter(MultiInputter):
  """An multi inputter that applies several transformation on the same data."""

  def __init__(self,
               inputters,
               reducer=ConcatReducer(),
               dropout=0.0):
    """Initializes a mixed inputter.

    Args:
      inputters: A list of :class:`opennmt.inputters.inputter.Inputter`.
      reducer: A :class:`opennmt.layers.reducer.Reducer` to merge all inputs.
      dropout: The probability to drop units in the merged inputs.
    """
    super(MixedInputter, self).__init__(inputters, reducer=reducer)
    self.dropout = dropout

  def get_length(self, features):
    return self.inputters[0].get_length(features)

  def make_dataset(self, data_file):
    return self.inputters[0].make_dataset(data_file)

  def get_dataset_size(self, data_file):
    return self.inputters[0].get_dataset_size(data_file)

  def _get_receiver_tensors(self):
    receiver_tensors = {}
    for inputter in self.inputters:
      receiver_tensors.update(inputter._get_receiver_tensors())  # pylint: disable=protected-access
    return receiver_tensors

  def make_features(self, element=None, features=None):
    if features is None:
      features = {}
    for inputter in self.inputters:
      features = inputter.make_features(element=element, features=features)
    return features

  def __call__(self, features, training=True):
    inputs = []
    for i, inputter in enumerate(self.inputters):
      with tf.variable_scope("inputter_{}".format(i)):
        inputs.append(inputter(features, training=training))
    inputs = self.reducer(inputs)
    inputs = tf.layers.dropout(inputs, rate=self.dropout, training=training)
    return inputs


def _extract_suffixed_keys(dictionary, suffix):
  """Returns a dictionary with all keys from :obj:`dictionary` that are suffixed
  with :obj:`suffix`.
  """
  sub_dict = {}
  for key, value in six.iteritems(dictionary):
    if key.endswith(suffix):
      original_key = key[:-len(suffix)]
      sub_dict[original_key] = value
  return sub_dict
