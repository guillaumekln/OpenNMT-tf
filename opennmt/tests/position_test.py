import tensorflow as tf
import numpy as np

from opennmt.layers import position


class _DummyPositionEncoder(position.PositionEncoder):
  """Encoder that simply forwards the position indices."""

  def encode(self, positions, depth):
    positions = tf.expand_dims(positions, 2)
    positions = tf.tile(positions, [1, 1, depth])
    return tf.cast(positions, self.dtype)


class PositionTest(tf.test.TestCase):

  def testApplyOneEncoding(self):
    encoder = _DummyPositionEncoder()
    inputs = tf.placeholder_with_default(np.zeros((2, 1, 3)), shape=(None, None, 3))
    outputs = encoder(inputs, position=2)
    with self.session() as sess:
      outputs = sess.run(outputs)
      self.assertAllEqual(outputs, [[[2, 2, 2]], [[2, 2, 2]]])

  def testApplyPositionEncoding(self):
    encoder = _DummyPositionEncoder()
    inputs = tf.placeholder_with_default(np.zeros((2, 4, 3)), shape=(None, None, 3))
    outputs = encoder(inputs)
    with self.session() as sess:
      outputs = sess.run(outputs)
      self.assertAllEqual(outputs, [
        [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
        [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]
      ])

  def testPositionEmbedder(self):
    encoder = position.PositionEmbedder()
    inputs = tf.placeholder_with_default(np.zeros((2, 4, 3)), shape=(None, None, 3))
    outputs = encoder(inputs)
    with self.session() as sess:
      sess.run(tf.global_variables_initializer())
      outputs = sess.run(outputs)

  def _testSinusoidalPositionEncoder(self, depth, dtype=tf.float32):
    encoder = position.SinusoidalPositionEncoder()
    inputs = tf.placeholder_with_default(
        np.zeros((2, 6, depth), dtype.as_numpy_dtype()), shape=(None, None, depth))
    outputs = encoder(inputs)
    self.assertEqual(dtype, inputs.dtype.base_dtype)
    with self.session() as sess:
      outputs = sess.run(outputs)
      self.assertAllEqual([2, 6, depth], outputs.shape)

  def testSinusoidalPositionEncoder(self):
    self._testSinusoidalPositionEncoder(10)
  def testSinusoidalPositionEncoderFloat16(self):
    self._testSinusoidalPositionEncoder(10, dtype=tf.float16)
  def testSinusoidalPositionEncoderInvalidDepth(self):
    with self.assertRaises(ValueError):
      self._testSinusoidalPositionEncoder(5)


if __name__ == "__main__":
  tf.test.main()
