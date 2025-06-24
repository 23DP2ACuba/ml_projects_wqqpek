"""
IMDB Reviews Sentiment alanisys
"""

import tensorflow_datasets as tfds
import tensorflow.keras 
import keras.layers
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import spacy

ds_train, ds_test = tfds.load('imdb_reviews', split=['train', 'test'], as_supervised=True)

nlp = spacy.load("en_core_web_sm")
def lemmatizatize_text(text):
  doc = nlp(text)
  return " ".join([token.lemma_ for token in doc])

def lemmatizatize_ds(df):
  text = []
  label = []
  for t, l in df:
    t = t.numpy().decode('utf-8')
    text.append(lemmatizatize_text(t))
    label.append(l.numpy())
  print("|", end="")
  return text, label

x_test, y_test = lemmatizatize_ds(ds_test)
x_train, y_train = lemmatizatize_ds(ds_train)

MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 250

vectorizer = TextVectorization(
    max_tokens=MAX_VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH,
    pad_to_max_tokens=True,
)

x_test = vectorizer.adapt(x_test)
x_train = vectorizer.adapt(x_train)
y_train = tf.constant(y_train)
y_test = tf.constant(y_test)

model = keras.Sequential(
    [
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="softmax"),
    ]
)

model.compile(
    loss = keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(3e-4),
    metrics = ["accuracy"]
)

model.fit(x_train, y_train, epochs=30, verbose=1)
model.evaluate(x_test, y_test)
