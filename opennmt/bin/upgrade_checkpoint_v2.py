"""Script to convert checkpoint to the V2 layout."""

import argparse

import tensorflow as tf

from opennmt.utils import checkpoint


def main():
  tf.logging.set_verbosity(tf.logging.INFO)

  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--model_dir", default=None,
                      help="The path to the model directory.")
  parser.add_argument("--checkpoint_path", default=None,
                      help="The path to the checkpoint to convert.")
  parser.add_argument("--output_dir", required=True,
                      help="The output directory where the updated checkpoint will be saved.")
  args = parser.parse_args()
  if args.model_dir is None and args.checkpoint_path is None:
    raise ValueError("One of --checkpoint_path and --model_dir should be set")
  checkpoint_path = args.checkpoint_path
  if checkpoint_path is None:
    checkpoint_path = tf.train.latest_checkpoint(args.model_dir)
  checkpoint.upgrade_checkpoint_to_v2(
      checkpoint_path,
      args.output_dir,
      session_config=tf.ConfigProto(device_count={"GPU": 0}))


if __name__ == "__main__":
  main()
