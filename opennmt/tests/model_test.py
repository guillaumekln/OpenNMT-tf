# -*- coding: utf-8 -*-

import os

from numbers import Number

import tensorflow as tf

from opennmt import constants, models, encoders, inputters
from opennmt.models import catalog
from opennmt.utils.vocab import Vocab


def _make_vocab_from_file(path, data_file):
  vocab = Vocab(special_tokens=[
      constants.PADDING_TOKEN,
      constants.START_OF_SENTENCE_TOKEN,
      constants.END_OF_SENTENCE_TOKEN])
  vocab.add_from_text(data_file)
  vocab.serialize(path)
  return path

def _make_data_file(path, lines):
  with open(path, "w") as data:
    for line in lines:
      data.write("%s\n" % line)
  return path


class ModelTest(tf.test.TestCase):

  def _testGenericModel(self,
                        model,
                        mode,
                        features_file,
                        labels_file,
                        metadata,
                        batch_size=16,
                        prediction_heads=None,
                        metrics=None,
                        params=None):
    # Mainly test that the code does not throw.
    if params is None:
      params = model.auto_config()["params"]
    dataset = model.input_fn(
        mode,
        batch_size,
        metadata,
        features_file,
        labels_file=labels_file if mode != tf.estimator.ModeKeys.PREDICT else None)()
    iterator = dataset.make_initializable_iterator()
    data = iterator.get_next()
    if mode != tf.estimator.ModeKeys.PREDICT:
      features, labels = data
    else:
      features, labels = data, None
    estimator_spec = model.model_fn()(features, labels, params, mode, None)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      sess.run(tf.tables_initializer())
      sess.run(iterator.initializer)
      if mode == tf.estimator.ModeKeys.TRAIN:
        loss = sess.run(estimator_spec.loss)
        self.assertIsInstance(loss, Number)
      elif mode == tf.estimator.ModeKeys.EVAL:
        fetches = [estimator_spec.loss]
        if estimator_spec.eval_metric_ops is not None:
          fetches.append(estimator_spec.eval_metric_ops)
        result = sess.run(fetches)
        self.assertIsInstance(result[0], Number)
        if metrics is not None:
          for metric in metrics:
            self.assertIn(metric, result[1])
      else:
        predictions = sess.run(estimator_spec.predictions)
        self.assertIsInstance(predictions, dict)
        if prediction_heads is not None:
          for head in prediction_heads:
            self.assertIn(head, predictions)

  def _makeToyEnDeData(self):
    metadata = {}
    features_file = _make_data_file(
        os.path.join(self.get_temp_dir(), "src.txt"),
        ["Parliament Does Not Support Amendment Freeing Tymoshenko",
         "Today , the Ukraine parliament dismissed , within the Code of Criminal Procedure "
         "amendment , the motion to revoke an article based on which the opposition leader , "
         "Yulia Tymoshenko , was sentenced .",
         "The amendment that would lead to freeing the imprisoned former Prime Minister was "
         "revoked during second reading of the proposal for mitigation of sentences for "
         "economic offences ."])
    labels_file = _make_data_file(
        os.path.join(self.get_temp_dir(), "tgt.txt"),
        ["Keine befreiende Novelle für Tymoshenko durch das Parlament",
         "Das ukrainische Parlament verweigerte heute den Antrag , im Rahmen einer Novelle "
         "des Strafgesetzbuches denjenigen Paragrafen abzuschaffen , auf dessen Grundlage die "
         "Oppositionsführerin Yulia Timoshenko verurteilt worden war .",
         "Die Neuregelung , die den Weg zur Befreiung der inhaftierten Expremierministerin hätte "
         "ebnen können , lehnten die Abgeordneten bei der zweiten Lesung des Antrags auf Milderung "
         "der Strafen für wirtschaftliche Delikte ab ."])
    metadata["source_vocabulary"] = _make_vocab_from_file(
        os.path.join(self.get_temp_dir(), "src_vocab.txt"), features_file)
    metadata["target_vocabulary"] = _make_vocab_from_file(
        os.path.join(self.get_temp_dir(), "tgt_vocab.txt"), labels_file)
    return features_file, labels_file, metadata

  def _testSequenceToSequence(self, mode):
    model = catalog.NMTSmall()
    features_file, labels_file, metadata = self._makeToyEnDeData()
    self._testGenericModel(
        model,
        mode,
        features_file,
        labels_file,
        metadata,
        prediction_heads=["tokens", "length", "log_probs"])

  def testSequenceToSequenceTraining(self):
    self._testSequenceToSequence(tf.estimator.ModeKeys.TRAIN)
  def testSequenceToSequenceEvaluation(self):
    self._testSequenceToSequence(tf.estimator.ModeKeys.EVAL)
  def testSequenceToSequenceInference(self):
    self._testSequenceToSequence(tf.estimator.ModeKeys.PREDICT)

  def testSequenceToSequenceServing(self):
    # Test that serving features can be forwarded into the model.
    model = catalog.NMTSmall()
    _, _, metadata = self._makeToyEnDeData()
    features = model.serving_input_fn(metadata)().features
    with tf.variable_scope(model.name):
      outputs = model(features, None, model.auto_config()["params"], tf.estimator.ModeKeys.PREDICT)
      self.assertIsInstance(outputs["predictions"], dict)

  def _makeToyClassifierData(self):
    metadata = {}
    features_file = _make_data_file(
        os.path.join(self.get_temp_dir(), "src.txt"),
        ["This product was not good at all , it broke on the first use !",
         "Perfect , it does everything I need .",
         "How do I change the battery ?"])
    labels_file = _make_data_file(
        os.path.join(self.get_temp_dir(), "labels.txt"), ["negative", "positive", "neutral"])
    metadata["source_vocabulary"] = _make_vocab_from_file(
        os.path.join(self.get_temp_dir(), "src_vocab.txt"), features_file)
    metadata["target_vocabulary"] = _make_data_file(
        os.path.join(self.get_temp_dir(), "labels_vocab.txt"), ["negative", "positive", "neutral"])
    return features_file, labels_file, metadata

  def _testSequenceClassifier(self, mode):
    model = models.SequenceClassifier(inputters.WordEmbedder(10), encoders.MeanEncoder())
    features_file, labels_file, metadata = self._makeToyClassifierData()
    params = {
        "optimizer": "GradientDescentOptimizer",
        "learning_rate": 0.1}
    self._testGenericModel(
        model,
        mode,
        features_file,
        labels_file,
        metadata,
        prediction_heads=["classes"],
        metrics=["accuracy"],
        params=params)

  def testSequenceClassifierTraining(self):
    self._testSequenceClassifier(tf.estimator.ModeKeys.TRAIN)
  def testSequenceClassifierEvaluation(self):
    self._testSequenceClassifier(tf.estimator.ModeKeys.EVAL)
  def testSequenceClassifierInference(self):
    self._testSequenceClassifier(tf.estimator.ModeKeys.PREDICT)

  def _makeToyTaggerData(self):
    metadata = {}
    features_file = _make_data_file(
        os.path.join(self.get_temp_dir(), "src.txt"),
        ["M . Smith went to Washington .",
         "I live in New Zealand ."])
    labels_file = _make_data_file(
        os.path.join(self.get_temp_dir(), "labels.txt"),
        ["B-PER I-PER E-PER O O S-LOC O",
         "O O O B-LOC E-LOC O"])
    metadata["source_vocabulary"] = _make_vocab_from_file(
        os.path.join(self.get_temp_dir(), "src_vocab.txt"), features_file)
    metadata["target_vocabulary"] = _make_data_file(
        os.path.join(self.get_temp_dir(), "labels_vocab.txt"),
        ["O", "B-LOC", "I-LOC", "E-LOC", "S-LOC", "B-PER", "I-PER", "E-PER", "S-PER"])
    return features_file, labels_file, metadata

  def _testSequenceTagger(self, mode):
    model = models.SequenceTagger(
        inputters.WordEmbedder(10),
        encoders.MeanEncoder(),
        crf_decoding=True,
        tagging_scheme="bioes")
    features_file, labels_file, metadata = self._makeToyTaggerData()
    params = {
        "optimizer": "GradientDescentOptimizer",
        "learning_rate": 0.1}
    self._testGenericModel(
        model,
        mode,
        features_file,
        labels_file,
        metadata,
        prediction_heads=["tags", "length"],
        metrics=["accuracy", "precision", "recall", "f1"],
        params=params)

  def testSequenceTaggerTraining(self):
    self._testSequenceTagger(tf.estimator.ModeKeys.TRAIN)
  def testSequenceTaggerEvaluation(self):
    self._testSequenceTagger(tf.estimator.ModeKeys.EVAL)
  def testSequenceTaggerInference(self):
    self._testSequenceTagger(tf.estimator.ModeKeys.PREDICT)


if __name__ == "__main__":
  tf.test.main()
