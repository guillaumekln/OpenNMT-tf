import tensorflow as tf
import numpy as np

from opennmt.layers import position


class _DummyPositionEncoder(position.PositionEncoder):
  """Encoder that simply forwards the position indices."""

  def encode(self, positions, depth, dtype=tf.float32):
    positions = tf.expand_dims(positions, 2)
    positions = tf.tile(positions, [1, 1, depth])
    return tf.cast(positions, dtype)


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

  def _testSinusoidalPositionEncoder(self, depth, dtype=tf.float32):
    encoder = position.SinusoidalPositionEncoder()
    positions = [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]
    encoding = encoder.encode(positions, depth, dtype=dtype)
    self.assertEqual(dtype, encoding.dtype.base_dtype)
    with self.session() as sess:
      encoding = sess.run(encoding)
      self.assertAllEqual([2, 6, depth], encoding.shape)

  def testSinusoidalPositionEncoder(self):
    self._testSinusoidalPositionEncoder(10)
  def testSinusoidalPositionEncoderFloat16(self):
    self._testSinusoidalPositionEncoder(10, dtype=tf.float16)
  def testSinusoidalPositionEncoderInvalidDepth(self):
    with self.assertRaises(ValueError):
      self._testSinusoidalPositionEncoder(5)


if __name__ == "__main__":
  tf.test.main()
