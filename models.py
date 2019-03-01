import tensorflow as tf

from data_utils import create_batch_tf_dataset

SEED = 15121


def _inference(images,
               model_type='vgg16',
               is_train=True,
               n_outputs=20):
  if model_type == 'naive':
    return _naive_inference(images, is_train=is_train, n_outputs=n_outputs)
  if model_type == 'my_model':
    with tf.variable_scope("my_model", reuse=tf.AUTO_REUSE):
      return _my_model(images, n_outputs=n_outputs)


def _naive_inference(images,
                     is_train=True,
                     n_outputs=20):
  """Naive model using only 1 FC layer, no dropout. For reference purpose.
  
    Validation and Test accuracy should be around 26%. 
  """
  H, W, C = (images.get_shape()[1].value,
             images.get_shape()[2].value,
             images.get_shape()[3].value)

  # create model parameters same for train, test, val
  with tf.variable_scope("naive", reuse=tf.AUTO_REUSE):
    w_soft = tf.get_variable("w", [H * W * C, n_outputs])

  images = tf.reshape(images, [-1, H * W * C])
  logits = tf.matmul(images, w_soft)

  return logits


def create_tf_ops(data_dict,
                  model_type='conv',
                  n_outputs=20,
                  init_lr=0.001,
                  l2_reg=1e-3,
                  batch_size=32):
  """ Create and finalize a TF graph including ops """
  dataset_dict = create_batch_tf_dataset(data_dict,
                                         batch_size=batch_size)
  train_dataset = dataset_dict["train"]
  val_dataset = dataset_dict["valid"]
  test_dataset = dataset_dict["test"]

  # for conciseness, this iterator will be shared between train, val, test
  # will switch the dataset when respective initializer is called first
  shared_iterator = tf.data.Iterator.from_structure(
    train_dataset.output_types,
    train_dataset.output_shapes
  )
  imgs, labels = shared_iterator.get_next()

  # Indicates whether we are in training or in test mode for inferennce graph
  is_training = tf.placeholder(tf.bool)

  # shared weights for inference as well
  logits = _inference(imgs,
                      model_type=model_type,
                      is_train=is_training,
                      n_outputs=n_outputs
                      )

  global_step = tf.Variable(0, dtype=tf.int32, trainable=False,
                            name="global_step")
  # loss function
  xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=labels)
  train_loss = tf.reduce_mean(xentropy)
  l2_loss = tf.losses.get_regularization_loss()
  train_loss += l2_reg * l2_loss

  # optimizer
  lr = tf.train.exponential_decay(init_lr, global_step * 64,
                                  50000, 0.98, staircase=True)
  optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)

  # train
  train_op = optimizer.minimize(train_loss, global_step=global_step)

  # predictions
  preds = tf.to_int32(tf.argmax(logits, axis=1))

  # top 5 predictions
  top5_preds = tf.nn.top_k(logits, k=5)

  # put everything into an ops dict
  ops = {
    "global_step": global_step,
    "is_training": is_training,
    "train_loss": train_loss,
    "preds": preds,
    "top5_preds": top5_preds,
    "labels": labels,
    "train_iterator": shared_iterator.make_initializer(train_dataset),
    "val_iterator": shared_iterator.make_initializer(val_dataset),
    "test_iterator": shared_iterator.make_initializer(test_dataset),
    "train_op": train_op,
  }
  return ops
