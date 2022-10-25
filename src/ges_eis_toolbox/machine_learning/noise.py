from copy import deepcopy
import numpy as np
from numpy.random import normal
from scipy.ndimage import gaussian_filter1d


def add_cartesian_normal_noise(
    dataset_X: np.ndarray,
    amplitude: float,
    is_polar: bool = False,
    is_percentage: bool = False
    ) -> np.ndarray:
    """
    Given an input dataset adds a normal noise to the real and imaginary part of each feature.

    Parameters
    ----------
    dataset_X: np.ndarray
        the array containing the input features.
    amplitude: float
        the amplitude of the normal noise. If is_percentage is set to False it encodes the
        absolute value on the noise variance else, is set to True, it ecodes the percentage 
        value of the fariance in respect to the maximum difference in spectrum amplitude
    is_polar: bool
        if set to True the input feature vector contains the polar representation of the impedance
    is_percentage: bool
        if set to True will use the provided amplitude value as the percentual of the difference
        between the maximum and minima or both the real and imaginary parts of the impedance
    
    Returns
    -------
    np.ndarray
        the output dataset with the added noise
    """

    # Create a local copy of the provided dataset
    X = deepcopy(dataset_X)

    # Evaluate the original shape of the input and, if necessary, change the shape to a 2D vector
    shape = X.shape
    if len(shape[1::]) == 1:
        X = X.reshape([shape[0], 2, int(shape[1] / 2)])

    # If the representation is polar change it to cartesian
    if is_polar:

        buffer = []
        for x in X:
            Z = x[0, :]*np.exp(1j*x[1, :])
            b = np.concatenate((Z.real, Z.imag), axis=0)
            b = b.reshape(x.shape)
            buffer.append(b)
        
        X = np.array(buffer)

    # Recompute the output by adding a normal noise to the real and imaginary parts
    output = []
    for x in X:

        if is_percentage:
            cart_max = max([max(x[0, :]), max(-x[1, :])])
            cart_min = min([min(x[0, :]), min(-x[1, :])])
            sigma = amplitude*(cart_max-cart_min)/100
        else:
            sigma = amplitude
            
        noise = normal(loc=0, scale=sigma, size=x.shape)
        x += noise
        output.append(x)
    
    output = np.array(output)

    # If the input was polar change back the real and imaginary parts to the polar form
    if is_polar:

        buffer = []
        for x in output:
            Z = x[0, :] + 1j*x[1, :]
            b = np.concatenate((np.absolute(Z), np.angle(Z)), axis=0)
            b = b.reshape(x.shape)
            buffer.append(b)
        
        output = np.array(buffer)

    # If necessary adjust back the shape of the array
    if output.shape != shape:
        output = output.reshape(shape)

    return output


def apply_cartesian_gaussian_filter(dataset_X: np.ndarray, sigma: float, is_polar: bool = False) -> np.ndarray:
    """
    Apply a gaussian filter to the real and imaginary parts of a dataset

    Parameters
    ----------
    dataset_X: np.ndarray
        the array containing the noisy input features.
    sigma: float
        the value of the standard deviation for Gaussian kernel used in the filtering operation
    is_polar: bool
        if set to True the input feature vector contains the polar representation of the impedance
    """
    # Create a local copy of the dataset given as input
    X = deepcopy(dataset_X)

    # Evaluate the original shape of the input and, if necessary, change the shape to a 2D vector
    shape = X.shape
    if len(shape[1::]) == 1:
        X = X.reshape([shape[0], 2, int(shape[1] / 2)])

    # If the representation is polar change it to cartesian
    if is_polar:

        buffer = []
        for x in X:
            Z = x[0, :]*np.exp(1j*x[1, :])
            b = np.concatenate((Z.real, Z.imag), axis=0)
            b = b.reshape(x.shape)
            buffer.append(b)
        
        X = np.array(buffer)

    # Apply a 1D gaussian filter to the real and imaginary parts of the spectrum
    output = []
    for x in X:

        out = np.concatenate((
                gaussian_filter1d(x[0, :], sigma=sigma),
                gaussian_filter1d(x[1, :], sigma=sigma)
            ),
            axis=0,
        )
        out = out.reshape(x.shape)

        output.append(out)
    
    output = np.array(output)

    # If the input was polar change back the real and imaginary parts to the polar form
    if is_polar:

        buffer = []
        for x in output:
            Z = x[0, :] + 1j*x[1, :]
            b = np.concatenate((np.absolute(Z), np.angle(Z)), axis=0)
            b = b.reshape(x.shape)
            buffer.append(b)
        
        output = np.array(buffer)

    # If necessary adjust back the shape of the array
    if output.shape != shape:
        output = output.reshape(shape)

    return output



    