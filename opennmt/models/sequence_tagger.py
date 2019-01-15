"""Sequence tagger."""

import tensorflow as tf
import numpy as np

from opennmt.models.model import Model
from opennmt.utils.misc import count_lines, print_bytes
from opennmt.utils.losses import cross_entropy_sequence_loss


class SequenceTagger(Model):
  """A sequence tagger."""

  def __init__(self,
               inputter,
               encoder,
               tagging_scheme=None,
               crf_decoding=False,
               daisy_chain_variables=False,
               name="seqtagger"):
    """Initializes a sequence tagger.

    Args:
      inputter: A :class:`opennmt.inputters.inputter.Inputter` to process the
        input data.
      encoder: A :class:`opennmt.encoders.encoder.Encoder` to encode the input.
      tagging_scheme: The tagging scheme used. For supported schemes (currently
        only BIOES), additional evaluation metrics could be computed such as
        precision, recall, etc.
      crf_decoding: If ``True``, add a CRF layer after the encoder.
      daisy_chain_variables: If ``True``, copy variables in a daisy chain
        between devices for this model. Not compatible with RNN based models.
      name: The name of this model.
    """
    super(SequenceTagger, self).__init__(
        name,
        features_inputter=inputter,
        daisy_chain_variables=daisy_chain_variables)

    self.encoder = encoder
    self.crf_decoding = crf_decoding
    if tagging_scheme:
      self.tagging_scheme = tagging_scheme.lower()
    else:
      self.tagging_scheme = None
    self.output_layer = None
    self.transition_params = None

  def _initialize(self, metadata, asset_dir=None):
    assets = super(SequenceTagger, self)._initialize(metadata, asset_dir=asset_dir)
    self.labels_vocabulary_file = metadata["target_vocabulary"]
    self.num_labels = count_lines(self.labels_vocabulary_file)
    self.output_layer = tf.keras.layers.Dense(self.num_labels)
    if self.crf_decoding:
      self.transition_params = tf.get_variable(
          "transitions", shape=[self.num_labels, self.num_labels])
    return assets

  def _get_labels_builder(self, labels_file):
    labels_vocabulary = tf.contrib.lookup.index_table_from_file(
        self.labels_vocabulary_file,
        vocab_size=self.num_labels)

    dataset = tf.data.TextLineDataset(labels_file)
    def _process_fn(x):
      tags = tf.strings.split([x]).values
      return {
          "tags": tags,
          "tags_id": labels_vocabulary.lookup(tags),
          "length": tf.shape(tags)[0]
      }
    return dataset, _process_fn

  def _call(self, features, labels, params, mode):
    length = self._get_features_length(features)
    inputs = self.features_inputter(features, training=mode == tf.estimator.ModeKeys.TRAIN)
    encoder_outputs, _, encoder_sequence_length = self.encoder(
        inputs,
        sequence_length=length,
        training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = self.output_layer(encoder_outputs)

    if mode != tf.estimator.ModeKeys.TRAIN:
      if self.crf_decoding:
        tags_id, _ = tf.contrib.crf.crf_decode(
            logits,
            self.transition_params,
            encoder_sequence_length)
        tags_id = tf.cast(tags_id, tf.int64)
      else:
        tags_prob = tf.nn.softmax(logits)
        tags_id = tf.argmax(tags_prob, axis=2)

      labels_vocab_rev = tf.contrib.lookup.index_to_string_table_from_file(
          self.labels_vocabulary_file,
          vocab_size=self.num_labels)

      # A tensor can not be both fed and fetched,
      # so identity a new tensor of "length" for export model to predict
      output_sequence_length = tf.identity(encoder_sequence_length)

      predictions = {
          "length": output_sequence_length,
          "tags_id": tags_id,
          "tags": labels_vocab_rev.lookup(tags_id)
      }
    else:
      predictions = None

    return {
        "logits": logits,
        "predictions": predictions
    }

  def compute_loss(self, outputs, labels, training=True, params=None):
    if params is None:
      params = {}
    if self.crf_decoding:
      log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
          outputs["logits"],
          tf.cast(labels["tags_id"], tf.int32),
          labels["length"],
          transition_params=self.transition_params)
      loss = tf.reduce_sum(-log_likelihood)
      loss_normalizer = tf.cast(tf.shape(log_likelihood)[0], loss.dtype)
      return loss, loss_normalizer
    else:
      return cross_entropy_sequence_loss(
          outputs["logits"],
          labels["tags_id"],
          labels["length"],
          label_smoothing=params.get("label_smoothing", 0.0),
          average_in_time=params.get("average_loss_in_time", False),
          training=training)

  def compute_metrics(self, predictions, labels):
    weights = tf.sequence_mask(
        labels["length"],
        maxlen=tf.shape(labels["tags"])[1],
        dtype=tf.float32)

    accuracy = tf.keras.metrics.Accuracy()
    accuracy.update_state(
        labels["tags_id"],
        predictions["tags_id"],
        sample_weight=weights)

    eval_metric_ops = {}
    eval_metric_ops["accuracy"] = accuracy

    if self.tagging_scheme in ("bioes",):
      flag_fn = None
      if self.tagging_scheme == "bioes":
        flag_fn = flag_bioes_tags

      gold_flags, predicted_flags = tf.py_function(
          flag_fn,
          [labels["tags"], predictions["tags"], labels["length"]],
          [tf.uint8, tf.uint8])

      precision_metric = tf.keras.metrics.Precision()
      recall_metric = tf.keras.metrics.Recall()

      precision_metric.update_state(gold_flags, predicted_flags)
      recall_metric.update_state(gold_flags, predicted_flags)

      precision = precision_metric.result()
      recall = recall_metric.result()
      f1 = (2 * precision * recall) / (recall + precision)

      eval_metric_ops["precision"] = precision_metric
      eval_metric_ops["recall"] = recall_metric
      eval_metric_ops["f1"] = (f1, tf.no_op())

    return eval_metric_ops

  def print_prediction(self, prediction, params=None, stream=None):
    tags = prediction["tags"][:prediction["length"]]
    sent = b" ".join(tags)
    print_bytes(sent, stream=stream)


def flag_bioes_tags(gold, predicted, sequence_length=None, dtype=np.uint8):
  """Flags chunk matches for the BIOES tagging scheme.

  This function will produce the gold flags and the predicted flags. For each aligned
  gold flag ``g`` and predicted flag ``p``:

  * when ``g == p == True``, the chunk has been correctly identified (true positive).
  * when ``g == False and p == True``, the chunk has been incorrectly identified (false positive).
  * when ``g == True and p == False``, the chunk has been missed (false negative).
  * when ``g == p == False``, the chunk has been correctly ignored (true negative).

  Args:
    gold: The gold tags as a Numpy 2D string array.
    predicted: The predicted tags as a Numpy 2D string array.
    sequence_length: The length of each sequence as Numpy array.
    dtype: The dtype of the output flags.

  Returns:
    A tuple ``(gold_flags, predicted_flags)``.
  """
  if isinstance(gold, tf.Tensor):
    gold = gold.numpy()
  if isinstance(predicted, tf.Tensor):
    predicted = predicted.numpy()
  if sequence_length is not None and isinstance(sequence_length, tf.Tensor):
    sequence_length = sequence_length.numpy()

  gold_flags = []
  predicted_flags = []

  def _add_true_positive():
    gold_flags.append(True)
    predicted_flags.append(True)
  def _add_false_positive():
    gold_flags.append(False)
    predicted_flags.append(True)
  def _add_true_negative():
    gold_flags.append(False)
    predicted_flags.append(False)
  def _add_false_negative():
    gold_flags.append(True)
    predicted_flags.append(False)

  def _match(ref, hyp, index, length):
    if ref[index].startswith(b"B"):
      match = True
      while index < length and not ref[index].startswith(b"E"):
        if ref[index] != hyp[index]:
          match = False
        index += 1
      match = match and index < length and ref[index] == hyp[index]
      return match, index
    return ref[index] == hyp[index], index

  for b in range(gold.shape[0]):
    length = sequence_length[b] if sequence_length is not None else gold.shape[1]

    # First pass to detect true positives and true/false negatives.
    index = 0
    while index < length:
      gold_tag = gold[b][index]
      match, index = _match(gold[b], predicted[b], index, length)
      if match:
        if gold_tag == b"O":
          _add_true_negative()
        else:
          _add_true_positive()
      else:
        if gold_tag != b"O":
          _add_false_negative()
      index += 1

    # Second pass to detect false postives.
    index = 0
    while index < length:
      pred_tag = predicted[b][index]
      match, index = _match(predicted[b], gold[b], index, length)
      if not match and pred_tag != b"O":
        _add_false_positive()
      index += 1

  return np.array(gold_flags, dtype=dtype), np.array(predicted_flags, dtype=dtype)
