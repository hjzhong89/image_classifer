import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import logging
from PIL import Image

logger = tf.get_logger().setLevel(logging.ERROR)


def load_model(model_path):
    """
    Loads a Keras model from an h5 file
    :param model_path:
    :return:
    """
    return tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})


def process_image(image):
    """
    Pre-process an image for the model by normalizing values and resizing to 224x224
    :param image:
    :return:
    """
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    return image.numpy()


def predict(image_path, model_path, top_k):
    model = load_model(model_path)
    image = np.expand_dims(process_image(np.asarray(Image.open(image_path))), axis=0)
    predictions = model.predict(image)
    probs, labels = tf.nn.top_k(predictions, top_k)
    probs = probs.numpy()[0]
    return probs, labels
