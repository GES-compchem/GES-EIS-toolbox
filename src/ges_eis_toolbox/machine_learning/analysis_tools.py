import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(
    model: tf.keras.models.Sequential,
    X: np.ndarray,
    y: np.ndarray,
    with_logits: bool = False,
) -> None:
    """
    Plots the confusion matrix representing the performance of a given classification 
    algorithm encoded in a given sequentual 'model'.

    Parameters
    ----------
    model: tf.keras.models.Sequential
        the sequential model encoding the trained neural network
    X: np.ndarray
        the features array to be feed to the model
    y: np.ndarray
        the label array associated to the X one
    with_logits: bool
        whether the output of the model is in logits form
    """

    yp = model.predict(X)

    if with_logits:
        yp = tf.nn.softmax(yp)

    yp = [np.argmax(y) for y in yp]

    cm = tf.math.confusion_matrix(y, yp)

    mat = plt.matshow(cm, cmap="Blues")
    for (i, j), z in np.ndenumerate(cm):
        plt.text(j, i, "{:d}".format(z), ha="center", va="center")

    plt.colorbar(mat)

    plt.show()

