from typing import Tuple
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

COLORS = ["blue", "red", "green", "purple", "brown", "black"]


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


def plot_dataset_bode(
    X: np.ndarray,
    Y: np.ndarray,
    only_label: int = None,
    is_polar: bool = True,
    label_based_colors: bool = False,
) -> None:
    """
    Generate the Bode plot of all the data in a given dataset.

    Parameters
    ----------
    X: np.ndarray
        features array containing the impedance data
    Y: np.ndarray
        lables array containing the class to which a given circuit belongs to
    only_label: int
        if set to a value different from None (default) will plot only the traces referred to the indicated
        lable class
    is_polar: bool
        if set to True (default) assumes the data to be composed by magnitude and phase of the impedance
        else will expect a feature vector containing the real an immaginary part of the impedance
    label_based_colors: bool
        if set to True uses a different color for each class type else, if set to False (default) will
        use a single color to represent all the data
    """

    s = X.shape

    if len(s[1::]) == 1:
        X = X.reshape([s[0], 2, int(s[1] / 2)])

    fig, (ax1, ax2) = plt.subplots(nrows=2)

    for x, y in zip(X, Y):

        if y != only_label and only_label is not None:
            continue

        color = COLORS[y % len(COLORS)] if label_based_colors else "blue"

        if is_polar:
            ax1.plot(x[0, :], c=color, alpha=0.2)
            ax2.plot(-x[1, :], c=color, alpha=0.2)
        else:
            Z = x[0, :] + 1j * x[1, :]
            ax1.plot(np.absolute(Z), c=color, alpha=0.2)
            ax2.plot(-np.angle(Z), c=color, alpha=0.2)

    ax1.set_ylabel("|Z|")
    ax2.set_ylabel(r"$-\varphi$")
    ax2.set_xlabel("index")

    plt.show()


def plot_dataset_nyquist(
    X: np.ndarray,
    Y: np.ndarray,
    only_label: int = None,
    is_polar: bool = True,
    label_based_colors: bool = False,
) -> None:
    """
    Generate the Nyquist plot of all the data in a given dataset.

    Parameters
    ----------
    X: np.ndarray
        features array containing the impedance data
    Y: np.ndarray
        lables array containing the class to which a given circuit belongs to
    only_label: int
        if set to a value different from None (default) will plot only the traces referred to the indicated
        lable class
    is_polar: bool
        if set to True (default) assumes the data to be composed by magnitude and phase of the impedance
        else will expect a feature vector containing the real an immaginary part of the impedance
    label_based_colors: bool
        if set to True uses a different color for each class type else, if set to False (default) will
        use a single color to represent all the data
    """

    s = X.shape

    if len(s[1::]) == 1:
        X = X.reshape([s[0], 2, int(s[1] / 2)])

    fig, ax = plt.subplots()

    for x, y in zip(X, Y):

        if y != only_label and only_label is not None:
            continue

        color = COLORS[y % len(COLORS)] if label_based_colors else "blue"

        if is_polar:
            Z = x[0, :] * np.exp(1j * x[1, :])
            ax.plot(np.real(Z), -np.imag(Z), c=color, alpha=0.2)
        else:
            ax.plot(x[0, :], -x[1, :], c=color, alpha=0.2)

    ax.set_ylabel("-Im(Z)")
    ax.set_xlabel("Re(Z)")

    plt.show()


def isolate_misclassified_examples(
    model: tf.keras.models.Sequential,
    X: np.ndarray,
    Y: np.ndarray,
    with_logits: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Isolates from a dataset the misclassified entries

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
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        a tuple containing the feature array of the misclassified example, their correct lable
        and the lable wrongfully attributed by the prediction procedure.
    """

    Yp = model.predict(X)

    if with_logits:
        Yp = tf.nn.softmax(Yp)

    Yp = [np.argmax(y) for y in Yp]

    X_miss, Y_miss, Yp_miss = [], [], []
    for x, y, yp in zip(X, Y, Yp):
        if y != yp:
            X_miss.append(x)
            Y_miss.append(y)
            Yp_miss.append(yp)

    return np.array(X_miss), np.array(Y_miss), np.array(Yp_miss)

