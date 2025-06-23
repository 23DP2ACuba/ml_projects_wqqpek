import tensorflow_datasets as tfds
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
