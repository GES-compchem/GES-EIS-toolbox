from typing import List, Union
import numpy
import numpy as np


class EIS_Spectrum:
    """
    Class to hold and manipulate electrochemical impedance spectroscopy spectra.

    Parameters
    ----------
    frequency: Union[numpy.ndarray, List[float]]
        the list of frequency values scanned during the experiment
    frequency: Union[numpy.ndarray, List[float]]
        the list of complex impedance values measured, for each frequency entry, during the experiment

    Raises
    ------
    TypeError
        exception raised when the wrong type of argument is passed to the class constructor
    ValueError
        exception raised when a mismatch is detected between the lengths of frequency and impedance
    """

    def __init__(
        self,
        frequency: Union[numpy.ndarray, List[float]],
        impedance: Union[numpy.ndarray, List[complex]],
    ) -> None:

        if type(frequency) != np.ndarray and type(frequency) != list:
            raise TypeError("frequency must be of type numpy.ndarray or List[float]")

        if type(impedance) != np.ndarray and type(impedance) != list:
            raise TypeError("impedance must be of type numpy.ndarray or List[float]")

        if len(frequency) != len(impedance):
            raise ValueError(
                "Mismatch between the number of frequency and impedance points."
            )

        self.__frequency: numpy.ndarray = np.array(frequency)
        self.__impedance: numpy.ndarray = np.array(impedance)

    @property
    def frequency(self) -> numpy.ndarray:
        """
        The list of frequency values scanned during the experiment

        Returns
        -------
        numpy.ndarray
            a numpy array containing each frequency point measured during the experiment
        """
        return self.__frequency

    @property
    def impedance(self) -> numpy.ndarray:
        """
        The list of impedance values measured during the experiment

        Returns
        -------
        numpy.ndarray
            a numpy array containing each complex impedance value measured during the experiment
        """
        return self.__impedance

    @property
    def real_Z(self) -> numpy.ndarray:
        """
        The list containing the real part of the impedance values measured during the experiment

        Returns
        -------
        numpy.ndarray
            a numpy array containing the real part of each impedance value measured during the experiment
        """
        return np.real(self.__impedance)

    @property
    def imag_Z(self) -> numpy.ndarray:
        """
        The list containing the imaginary part of the impedance values measured during the experiment

        Returns
        -------
        numpy.ndarray
            a numpy array containing the imaginary part of each impedance value measured during the experiment
        """
        return np.imag(self.__impedance)

    @property
    def norm_Z(self) -> numpy.ndarray:
        """
        The list containing the modulus of the impedance values measured during the experiment

        Returns
        -------
        numpy.ndarray
            a numpy array containing the modulus of each impedance value measured during the experiment
        """
        return np.absolute(self.__impedance)

    @property
    def phi_Z(self) -> numpy.ndarray:
        """
        The list containing the polar part/phase of the impedance values measured during the experiment

        Returns
        -------
        numpy.ndarray
            a numpy array containing the polar part/phase of each impedance value measured during the experiment
        """
        return np.angle(self.__impedance)
