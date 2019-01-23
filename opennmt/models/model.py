"""Base class for models."""

from __future__ import print_function

import abc
import six

import tensorflow as tf

from opennmt.utils import data
from opennmt.utils.misc import item_or_tuple


@six.add_metaclass(abc.ABCMeta)
class Model(object):
  """Base class for models."""

  def __init__(self,
               name,
               features_inputter=None,
               labels_inputter=None,
               dtype=None):
    self.name = name
    self.features_inputter = features_inputter
    self.labels_inputter = labels_inputter
    if self.labels_inputter is not None:
      self.labels_inputter.is_target = True
    if dtype is None and self.features_inputter is not None:
      self.dtype = features_inputter.dtype
    else:
      self.dtype = dtype or tf.float32

  def auto_config(self):
    """Returns automatic configuration values specific to this model.

    Returns:
      A partial training configuration.
    """
    return {}

  def __call__(self, features, labels, params, mode):
    """Calls the model function.

    Args:
      features: A dictionary of input features.
      labels: A dictionary of target labels.
      params: A dictionary of hyperparameters.
      mode: A ``tf.estimator.ModeKeys`` mode.

    Returns:
      A dictionary of model outputs.
    """
    with tf.variable_scope(self.name, initializer=self._initializer(params)):
      return self._call(features, labels, params, mode)

  def _initializer(self, params):
    """Returns the global initializer for this model.

    Args:
      params: A dictionary of hyperparameters.

    Returns:
      The initializer.
    """
    param_init = params.get("param_init")
    if param_init is not None:
      return tf.random_uniform_initializer(
          minval=-param_init, maxval=param_init, dtype=self.dtype)
    return None

  @abc.abstractmethod
  def _call(self, features, labels, params, mode):
    """Calls the model function.

    Args:
      features: A dictionary of input features.
      labels: A dictionary of target labels.
      params: A dictionary of hyperparameters.
      mode: A ``tf.estimator.ModeKeys`` mode.

    Returns:
      A dictionary of model outputs.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def compute_loss(self, outputs, labels, training=True, params=None):
    """Computes the loss.

    Args:
      outputs: The model outputs.
      labels: The dict of labels ``tf.Tensor``.
      training: Training mode.
      params: An optional dictionary of hyperparameters.

    Returns:
      The loss or a tuple containing the computed loss and the loss to display.
    """
    raise NotImplementedError()

  def compute_metrics(self, predictions, labels):  # pylint: disable=unused-argument
    """Computes additional metrics on the predictions.

    Args:
      labels: The dict of labels ``tf.Tensor``.
      predictions: The model predictions.

    Returns:
      A dict of metrics. See the ``eval_metric_ops`` field of
      ``tf.estimator.EstimatorSpec``.
    """
    return None

  def initialize(self, metadata):
    """Initializes the model from the data configuration.

    Args:
      metadata: A dictionary containing additional metadata set
        by the user.
    """
    if self.features_inputter is not None:
      self.features_inputter.initialize(metadata, asset_prefix="source_")
    if self.labels_inputter is not None:
      self.labels_inputter.initialize(metadata, asset_prefix="target_")

  def export_assets(self, asset_dir):
    """Exports assets used by this model.

    Args:
      asset_dir: The directory where assets can be written.

    Returns:
      A dictionary containing additional assets used by the model.
    """
    assets = {}
    if self.features_inputter is not None:
      assets.update(self.features_inputter.export_assets(asset_dir, asset_prefix="source_"))
    if self.labels_inputter is not None:
      assets.update(self.labels_inputter.export_assets(asset_dir, asset_prefix="target_"))
    return assets

  def get_serving_input_receiver(self):
    """Returns an input receiver for serving this model.

    Returns:
      A ``tf.estimator.export.ServingInputReceiver``.
    """
    if self.features_inputter is None:
      raise NotImplementedError()
    return self.features_inputter.get_serving_input_receiver()

  def _get_features_length(self, features):
    """Returns the features length.

    Args:
      features: A dict of ``tf.Tensor``.

    Returns:
      The length as a ``tf.Tensor`` or list of ``tf.Tensor``, or ``None`` if
      length is undefined.
    """
    if self.features_inputter is None:
      return None
    return self.features_inputter.get_length(features)

  def _get_labels_length(self, labels):
    """Returns the labels length.

    Args:
      labels: A dict of ``tf.Tensor``.

    Returns:
      The length as a ``tf.Tensor``  or ``None`` if length is undefined.
    """
    if self.labels_inputter is None:
      return None
    return self.labels_inputter.get_length(labels)

  def _get_dataset_size(self, features_file):
    """Returns the size of the dataset.

    Args:
      features_file: The file of features.

    Returns:
      The total size.
    """
    if self.features_inputter is None:
      raise NotImplementedError()
    return self.features_inputter.get_dataset_size(features_file)

  def _get_features_builder(self, features_file):
    """Returns the recipe to build features.

    Args:
      features_file: The file of features.

    Returns:
      A tuple ``(tf.data.Dataset, process_fn)``.
    """
    if self.features_inputter is None:
      raise NotImplementedError()
    dataset = self.features_inputter.make_dataset(features_file)
    process_fn = self.features_inputter.make_features
    return dataset, process_fn

  def _get_labels_builder(self, labels_file):
    """Returns the recipe to build labels.

    Args:
      labels_file: The file of labels.

    Returns:
      A tuple ``(tf.data.Dataset, process_fn)``.
    """
    if self.labels_inputter is None:
      raise NotImplementedError()
    dataset = self.labels_inputter.make_dataset(labels_file)
    process_fn = self.labels_inputter.make_features
    return dataset, process_fn

  def _augment_parallel_dataset(self, dataset, process_fn, mode=None):
    """Augments a parallel dataset.

    Args:
      dataset: A parallel dataset.
      process_fn: The current dataset processing function.
      mode: A ``tf.estimator.ModeKeys`` mode.

    Returns:
      A tuple ``(tf.data.Dataset, process_fn)``.
    """
    _ = mode
    return dataset, process_fn

  def make_dataset(self,
                   mode,
                   batch_size,
                   features_file,
                   labels_file=None,
                   batch_type="examples",
                   bucket_width=None,
                   single_pass=False,
                   num_threads=None,
                   shuffle_buffer_size=None,
                   maximum_features_length=None,
                   maximum_labels_length=None):
    """Returns a dataset for this model.

    Args:
      mode: A ``tf.estimator.ModeKeys`` mode.
      batch_size: The batch size to use.
      features_file: The file containing input features.
      labels_file: The file containing output labels.
      batch_type: The training batching stragety to use: can be "examples" or
        "tokens".
      bucket_width: The width of the length buckets to select batch candidates
        from. ``None`` to not constrain batch formation.
      single_pass: If ``True``, makes a single pass over the training data.
      num_threads: The number of elements processed in parallel.
      shuffle_buffer_size: Shuffle this many consecutive examples from the
        dataset.
      maximum_features_length: The maximum length or list of maximum lengths of
        the features sequence(s). ``None`` to not constrain the length.
      maximum_labels_length: The maximum length of the labels sequence.
        ``None`` to not constrain the length.

    Returns:
      A ``tf.data.Dataset``.

    Raises:
      ValueError: if :obj:`labels_file` is not set when in training or
        evaluation mode.
    """
    feat_dataset, feat_process_fn = self._get_features_builder(features_file)

    if labels_file is None:
      if mode != tf.estimator.ModeKeys.PREDICT:
        raise ValueError("Labels file is required for training and evaluation")
      dataset = feat_dataset
      # Parallel inputs must be catched in a single tuple and not considered as multiple arguments.
      process_fn = lambda *arg: feat_process_fn(item_or_tuple(arg))
    else:
      labels_dataset, labels_process_fn = self._get_labels_builder(labels_file)

      dataset = tf.data.Dataset.zip((feat_dataset, labels_dataset))
      process_fn = lambda features, labels: (
          feat_process_fn(features), labels_process_fn(labels))
      dataset, process_fn = self._augment_parallel_dataset(dataset, process_fn, mode=mode)

    if mode == tf.estimator.ModeKeys.TRAIN:
      batch_size_multiple = 1
      if batch_type == "tokens" and self.dtype == tf.float16:
        batch_size_multiple = 8
      dataset = data.training_pipeline(
          dataset,
          batch_size,
          batch_type=batch_type,
          bucket_width=bucket_width,
          single_pass=single_pass,
          process_fn=process_fn,
          num_threads=num_threads,
          shuffle_buffer_size=shuffle_buffer_size,
          dataset_size=self._get_dataset_size(features_file),
          maximum_features_length=maximum_features_length,
          maximum_labels_length=maximum_labels_length,
          features_length_fn=self._get_features_length,
          labels_length_fn=self._get_labels_length,
          batch_size_multiple=batch_size_multiple)
    else:
      dataset = data.inference_pipeline(
          dataset,
          batch_size,
          process_fn=process_fn,
          num_threads=num_threads,
          bucket_width=bucket_width,
          length_fn=self._get_features_length)

    return dataset

  def print_prediction(self, prediction, params=None, stream=None):
    """Prints the model prediction.

    Args:
      prediction: The evaluated prediction.
      params: (optional) Dictionary of formatting parameters.
      stream: (optional) The stream to print to.
    """
    _ = params
    print(prediction, file=stream)
