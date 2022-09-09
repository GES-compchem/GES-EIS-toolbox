import numpy
import numpy as np


class EIS_Spectrum:

    def __init__(self, frequency, impedance) -> None:
        self.__frequency = frequency
        self.__impedance = impedance
    
    @property
    def frequency(self) -> numpy.ndarray:
        return self.__frequency
    
    @property
    def impedance(self) -> numpy.ndarray:
        return self.__impedance
