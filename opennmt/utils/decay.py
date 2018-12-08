"""Define learning rate decay functions."""

import tensorflow as tf


# All functions must take the learning rate and the step as first arguments.

def noam_decay(scale, step, model_dim, warmup_steps):
  """Defines the decay function described in https://arxiv.org/abs/1706.03762.

  Args:
    scale: The scale constant.
    step: The current step.
    model_dim: The model dimension.
    warmup_steps: The number of warmup steps.

  Returns:
    The learning rate for the step :obj:`global_step`.
  """
  step = tf.cast(step + 1, tf.float32)
  model_dim = tf.cast(model_dim, tf.float32)
  warmup_steps = tf.cast(warmup_steps, tf.float32)
  return (scale
          * tf.pow(model_dim, -0.5)
          * tf.minimum(tf.pow(step, -0.5), step * tf.pow(warmup_steps, -1.5)))


def rsqrt_decay(scale, step, warmup_steps):
  """Decay based on the reciprocal of the step square root.

  Args:
    scale: The scale constant.
    step: The current step.
    warmup_steps: The number of warmup steps.

  Returns:
    The learning rate for the step :obj:`global_step`.
  """
  step = tf.cast(step, tf.float32)
  warmup_steps = tf.cast(warmup_steps, tf.float32)
  return scale * tf.rsqrt(tf.maximum(step, warmup_steps))


def rnmtplus_decay(scale,
                   step,
                   num_replicas,
                   warmup_steps=500,
                   start_step=600000,
                   end_step=1200000):
  """Defines the decay function described in https://arxiv.org/abs/1804.09849.

  Args:
    scale: The scale constant.
    step: The current step.
    num_replicas: The number of concurrent model replicas.
    warmup_steps: The number of warmup steps.
    start_step: The start step of the exponential decay.
    end_step: The end step of the exponential decay.

  Returns:
    The learning rate for the step :obj:`step`.
  """
  t = tf.cast(step, tf.float32)
  n = tf.cast(num_replicas, tf.float32)
  p = tf.cast(warmup_steps, tf.float32)
  s = tf.cast(start_step, tf.float32)
  e = tf.cast(end_step, tf.float32)
  return scale * tf.minimum(
      tf.minimum(1 + (t * (n - 1)) / (n * p), n),
      n * tf.pow(2 * n, (s - n * t) / (e - s)))
