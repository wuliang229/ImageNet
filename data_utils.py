from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections import OrderedDict

import tensorflow as tf

N_CLASSES_LIMIT = 50


def parse_data(dir_path='tiny-imagenet-200', n_classes_limit=N_CLASSES_LIMIT):
  """dir_path must be accessible from the location running this script"""

  def _read_wnids(filename='wnids.txt'):
    wnids_file = os.path.join(dir_path, filename)
    assert os.path.exists(wnids_file)
    wnids = OrderedDict()
    c = 0
    with open(wnids_file) as f:
      for line in f:
        wnids[line.strip()] = c
        c += 1

        # only take 50 labels
        if c == n_classes_limit:
          break

    print("%d labels" % c)
    return wnids

  def _enumerate_train_imgs(train_dir, wnids):
    assert os.path.isdir(train_dir)
    subdirs = os.listdir(train_dir)

    train_imgs, train_labels = [], []
    test_imgs, test_labels = [], []

    for subdir in subdirs:
      if subdir != ".DS_Store": # On Mac only
        c = 0  # for each subdir (i.e. label), take first 50 samples for test
        subdir_path = os.path.join(train_dir, subdir, "images")
        imgs = os.listdir(subdir_path)

        try:
          label_idx = wnids[subdir]
        except:
          continue

        for img in imgs:
          full_img_path = os.path.join(subdir_path, img)
          if not os.path.exists(full_img_path):
            print("%s not existed ..." % full_img_path)
            continue

          if c < 50:
            test_imgs.append(full_img_path)
            test_labels.append(label_idx)
          else:
            train_imgs.append(full_img_path)
            train_labels.append(label_idx)

          c += 1

    return train_imgs, train_labels, test_imgs, test_labels

  def _read_val_annotations(val_annotation, wnids):
    val_imgs, val_labels = [], []
    with open(val_annotation) as f:
      for line in f:
        fields = line.strip().split()
        img, label = fields[0], fields[1]
        full_img_path = os.path.join(dir_path, "val", "images", img)
        if not os.path.exists(full_img_path):
          print("%s not existed ..." % full_img_path)
          continue

        try:
          label_idx = wnids[label]
        except:
          continue

        val_imgs.append(full_img_path)
        val_labels.append(label_idx)

    return val_imgs, val_labels

  wnids = _read_wnids()
  train_dir = os.path.join(dir_path, "train")
  val_annotation = os.path.join(dir_path, "val", "val_annotations.txt")

  # take train and test images
  train_imgs, train_labels, test_imgs, test_labels = \
    _enumerate_train_imgs(train_dir, wnids)

  # take val images
  val_imgs, val_labels = _read_val_annotations(val_annotation, wnids)

  data_dict = {
    "train_imgs": train_imgs,
    "train_labels": train_labels,
    "val_imgs": val_imgs,
    "val_labels": val_labels,
    "test_imgs": test_imgs,
    "test_labels": test_labels
  }

  return data_dict # A dictionary of lists of data paths and labels


def _pre_process(img, label):
  """Process image only, input should be Tensors. Applied for train dataset only. 
  """
  img = tf.pad(img, [[4, 4], [4, 4], [0, 0]])
  img = tf.random_crop(img, [56, 56, 3])
  img = tf.image.random_flip_left_right(img)
  img = tf.image.random_brightness(img, max_delta=63)
  img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
  return img, label


def _parse_function(filename, label):
  """Read filepath into TF Tensors. Applied for all train, val, test datasets 
  """
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string, channels=3)
  image_decoded = tf.cast(image_decoded, tf.float32)
  image_resized = tf.image.resize_image_with_crop_or_pad(image_decoded, 56, 56)

  return image_resized, label


def create_batch_tf_dataset(data_dict,
                            batch_size=32,
                            n_workers=10,
                            buffer_size=10000,
                            ):
  """ Similar to CIFAR10, create batched datasets for train, val, 
    and test.

    data_dict is a dictionary of lists of data
  """


  # data augmentation is needed for train dataset
  train_imgs = tf.constant(data_dict["train_imgs"])
  train_labels = tf.constant(data_dict["train_labels"])
  train_dataset = tf.data.Dataset.from_tensor_slices((train_imgs, train_labels))
  train_dataset = train_dataset.map(_parse_function, num_parallel_calls=n_workers)
  train_dataset = train_dataset.map(_pre_process, num_parallel_calls=n_workers)
  batched_train_dataset = train_dataset.repeat().shuffle(buffer_size).batch(batch_size)


  # val dataset
  val_imgs = tf.constant(data_dict["val_imgs"])
  val_labels = tf.constant(data_dict["val_labels"])
  val_dataset = tf.data.Dataset.from_tensor_slices((val_imgs, val_labels))
  val_dataset = val_dataset.map(_parse_function, num_parallel_calls=n_workers)
  batched_val_dataset = val_dataset.batch(batch_size)

  # test dataset
  test_imgs = tf.constant(data_dict["test_imgs"])
  test_labels = tf.constant(data_dict["test_labels"])
  test_dataset = tf.data.Dataset.from_tensor_slices((test_imgs, test_labels))
  test_dataset = test_dataset.map(_parse_function, num_parallel_calls=n_workers)
  batched_test_dataset = test_dataset.batch(batch_size)

  return {
    "train": batched_train_dataset,
    "valid": batched_val_dataset,
    "test": batched_test_dataset
  }


if __name__ == "__main__":
  data_dict = parse_data()
  dataset_dict = create_batch_tf_dataset(data_dict)
  import pdb;

  pdb.set_trace()
