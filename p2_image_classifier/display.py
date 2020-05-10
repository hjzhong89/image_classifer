from PIL import Image
import numpy as np
import json
from matplotlib import pyplot as plt


def get_category_names(labels, category_names, offset=1):
    """
    Return the class names from the target labels
    :param labels: Labels to fetch class names for
    :param category_names: File path to JSON class names
    :param offset: Offset if category_names dictionary is not 0 based.
    :return:
    """
    with open(category_names, 'r') as f:
        category_names = json.load(f)

    labels = [category_names[str(i + offset)] for i in labels.numpy()[0]]
    return labels


def display_results(image_path, probs, labels, category_names):
    """
    Display the results from the classifier alongside the image
    :param image_path: File path to the image being classified
    :param probs: Probabilities of prediction
    :param labels: Labels returned by classifier
    :param category_names: File path to JSON category names
    :return:
    """
    labels = get_category_names(labels, category_names)
    plt.figure(figsize=(8, 8))

    test_image = np.asarray(Image.open(image_path))
    plt.subplot(1, 2, 1)
    plt.imshow(test_image)
    plt.title(image_path)

    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(get_category_names(labels, category_names)))
    plt.barh(y_pos, probs, tick_label=labels)
