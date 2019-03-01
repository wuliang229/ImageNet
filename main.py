from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import sys

import tensorflow as tf
import numpy as np

from data_utils import parse_data
from models import create_tf_ops
from utils import DEFINE_boolean
from utils import DEFINE_integer
from utils import DEFINE_string
from utils import DEFINE_float
from utils import print_user_flags

from data_utils import N_CLASSES_LIMIT # Different from CIFAR-10 main.py

flags = tf.app.flags
FLAGS = flags.FLAGS

DEFINE_boolean("reset_output_dir", False, "Delete output_dir if exists.")
DEFINE_string("data_path", "", "Path to CIFAR-10 data")
DEFINE_string("output_dir", "output", "Path to log folder")
DEFINE_string("model_name", "conv",
              "Name of the model.")
DEFINE_integer("n_epochs", 50, "How many epochs to run in total")
DEFINE_integer("train_steps", 700, "How many batches per epoch")
DEFINE_integer("log_every", 700, "How many steps to log")
DEFINE_integer("n_classes", N_CLASSES_LIMIT, "Number of classes")
DEFINE_integer("batch_size", 32, "Batch size")
DEFINE_float("init_lr", 1e-3, "Init learning rate")


def get_ops(data_dict):
  """Builds the model."""
  print("-" * 80)
  print("Creating a '{0}' model".format(FLAGS.model_name))

  ops = create_tf_ops(data_dict,
                      model_type=FLAGS.model_name,
                      n_outputs=FLAGS.n_classes,
                      init_lr=FLAGS.init_lr,
                      batch_size=FLAGS.batch_size)

  assert "global_step" in ops
  assert "train_op" in ops
  assert "train_loss" in ops

  return ops


def main(_):
  print("-" * 80)
  if not os.path.isdir(FLAGS.output_dir):
    print("Path {0} does not exist. Creating.".format(FLAGS.output_dir))
    os.makedirs(FLAGS.output_dir)
  elif FLAGS.reset_output_dir:
    print("Path {0} exists. Remove and remake.".format(FLAGS.output_dir))
    shutil.rmtree(FLAGS.output_dir)
    os.makedirs(FLAGS.output_dir)

  print_user_flags()

  # data
  data_dict = parse_data()

  # computational graph
  g = tf.Graph()
  tf.reset_default_graph()

  with g.as_default():
    ops = get_ops(data_dict)

    print("-" * 80)

    print("Starting session")
    config = tf.ConfigProto(allow_soft_placement=True)

    # hook up with a session to train
    with tf.train.SingularMonitoredSession(
        config=config, checkpoint_dir=FLAGS.output_dir) as sess:

      # training loop
      print("-" * 80)
      print("Starting training")

      for epoch in range(1, FLAGS.n_epochs + 1):
        sess.run(ops["train_iterator"])  # init dataset iterator
        for step in range(1, FLAGS.train_steps + 1):
          
          # Run respective ops for each training step
          labels, loss, _, preds = sess.run([ops["labels"], ops["train_loss"], ops["train_op"], ops["preds"]], feed_dict = {ops["is_training"]: True})
          # sess.run(ops["train_op"], )

          if step > 0 and step % 100 == 0:
            acc = np.sum(preds == labels) / len(labels)
            print("Epoch %d Batch %d: loss = %.3f train_accuracy = %.3f" %
                  (epoch, step, loss, acc))

          if step % FLAGS.log_every == 0:
            # this will reset train_dataset as well, doesn't matter as long as FLAGS.log_every == FLAGS.train_steps
            get_eval_accuracy(ops, sess, step, "val")

      print("-" * 80)
      print("Training done. Eval on TEST set")
      get_eval_accuracy(ops, sess, step, "test")


def get_eval_accuracy(ops, sess, step, name="val"):
  if name == "val":
    sess.run(ops["val_iterator"])
  else:
    sess.run(ops["test_iterator"])

  preds_ops = ops["preds"]
  labels = ops["labels"]

  # Get accuracy for the whole dataset
  n_samples = 0
  total_val_acc = 0.0

  while True:
    try:
      preds, pred_labels = sess.run([preds_ops, labels], feed_dict = {ops["is_training"]: False})
      n_samples += len(pred_labels)
      total_val_acc += np.sum(preds == pred_labels)
    except tf.errors.OutOfRangeError:
      break

  total_val_acc /= n_samples

  log_string = "\n"
  log_string += "step={0:<6d}".format(step)
  log_string += " acc={0:.3f} against {1:<3d} samples\n".format(
    total_val_acc, n_samples)
  print(log_string)
  sys.stdout.flush()

if __name__ == "__main__":
  tf.app.run()
