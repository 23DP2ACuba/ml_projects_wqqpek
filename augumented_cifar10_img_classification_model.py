import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras.layers import LeakyReLU


(ds_train, ds_test), ds_info = tfds.load(
    "cifar10",
    split = ["train", "test"],
    shuffle_files = True,
    as_supervised = True,
    with_info = True
)


def normalize_img(image, label):
  return tf.cast(image, tf.float32) / 255.0, label


def augument(image, label):
  new_height = new_width = 32
  image = tf.image.resize(image, (new_height, new_width))

  if tf.random.uniform((), minval = 0, maxval = 1)<0.1:
    image = tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3])

  image = tf.image.random_contrast(image, lower=.1, upper=.2)
  image = tf.image.random_flip_left_right(image)

  return image, label

data_augumentation = keras.Sequential(
    [
        layers.Resizing(height=32, width=32),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1)
    ]
)

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32

ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
#ds_train = ds_train.map(augument, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.prefetch(AUTOTUNE)


model = keras.Sequential(
    [
        keras.Input(shape=(32,32,3)),
        data_augumentation,
        layers.Conv2D(64, 3, padding='valid', activation=LeakyReLU(alpha=0.05)),
        layers.Conv2D(128, 3, padding='valid', activation=LeakyReLU(alpha=0.05)),
        layers.Dropout(0.20),
        layers.MaxPooling2D(),
        layers.Conv2D(256, 3, padding='valid', activation=LeakyReLU(alpha=0.05)),
        layers.Flatten(),
        layers.Dense(1024, activation=LeakyReLU(alpha=0.05)),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax'),
    ]
)

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(),
    optimizer = keras.optimizers.Adam(3e-4),
    metrics = ["accuracy"]
)

model.fit(ds_train, epochs=10, verbose=1)
model.evaluate(ds_test)
