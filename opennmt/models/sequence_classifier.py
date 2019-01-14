"""Sequence classifier."""

import tensorflow as tf

from opennmt.models.model import Model
from opennmt.utils.cell import last_encoding_from_state
from opennmt.utils.misc import count_lines, print_bytes
from opennmt.utils.losses import cross_entropy_loss


class SequenceClassifier(Model):
  """A sequence classifier."""

  def __init__(self,
               inputter,
               encoder,
               encoding="average",
               daisy_chain_variables=False,
               name="seqclassifier"):
    """Initializes a sequence classifier.

    Args:
      inputter: A :class:`opennmt.inputters.inputter.Inputter` to process the
        input data.
      encoder: A :class:`opennmt.encoders.encoder.Encoder` to encode the input.
      encoding: "average" or "last" (case insensitive), the encoding vector to
        extract from the encoder outputs.
      daisy_chain_variables: If ``True``, copy variables in a daisy chain
        between devices for this model. Not compatible with RNN based models.
      name: The name of this model.

    Raises:
      ValueError: if :obj:`encoding` is invalid.
    """
    super(SequenceClassifier, self).__init__(
        name,
        features_inputter=inputter,
        daisy_chain_variables=daisy_chain_variables)

    self.encoder = encoder
    self.encoding = encoding.lower()
    if self.encoding not in ("average", "last"):
      raise ValueError("Invalid encoding vector: {}".format(self.encoding))
    self.output_layer = None

  def _initialize(self, metadata, asset_dir=None):
    assets = super(SequenceClassifier, self)._initialize(metadata, asset_dir=asset_dir)
    self.labels_vocabulary_file = metadata["target_vocabulary"]
    self.num_labels = count_lines(self.labels_vocabulary_file)
    self.output_layer = tf.keras.layers.Dense(self.num_labels)
    return assets

  def _get_labels_builder(self, labels_file):
    labels_vocabulary = tf.contrib.lookup.index_table_from_file(
        self.labels_vocabulary_file,
        vocab_size=self.num_labels)

    dataset = tf.data.TextLineDataset(labels_file)
    process_fn = lambda x: {
        "classes": x,
        "classes_id": labels_vocabulary.lookup(x)
    }
    return dataset, process_fn

  def _call(self, features, labels, params, mode):
    inputs = self.features_inputter(features, training=mode == tf.estimator.ModeKeys.TRAIN)
    encoder_outputs, encoder_state, _ = self.encoder(
        inputs,
        sequence_length=self._get_features_length(features),
        training=mode == tf.estimator.ModeKeys.TRAIN)

    if self.encoding == "average":
      encoding = tf.reduce_mean(encoder_outputs, axis=1)
    elif self.encoding == "last":
      encoding = last_encoding_from_state(encoder_state)

    logits = self.output_layer(encoding)

    if mode != tf.estimator.ModeKeys.TRAIN:
      labels_vocab_rev = tf.contrib.lookup.index_to_string_table_from_file(
          self.labels_vocabulary_file,
          vocab_size=self.num_labels)
      classes_prob = tf.nn.softmax(logits)
      classes_id = tf.argmax(classes_prob, axis=1)
      predictions = {
          "classes_id": classes_id,
          "classes": labels_vocab_rev.lookup(classes_id)
      }
    else:
      predictions = None

    return {
        "logits": logits,
        "predictions": predictions
    }

  def _compute_loss(self, features, labels, outputs, params, mode):
    return cross_entropy_loss(
        outputs["logits"],
        labels["classes_id"],
        label_smoothing=params.get("label_smoothing", 0.0),
        mode=mode)

  def _compute_metrics(self, features, labels, predictions):
    accuracy = tf.keras.metrics.Accuracy()
    accuracy.update_state(labels["classes_id"], predictions["classes_id"])
    return {
        "accuracy": accuracy
    }

  def print_prediction(self, prediction, params=None, stream=None):
    print_bytes(prediction["classes"], stream=stream)
