from enum import Enum
from os.path import isfile
from impedance.preprocessing import readGamry, readBioLogic

from ges_eis_toolbox.exceptions import FileNotFound, UnknownFileExtension
from ges_eis_toolbox.spectra.spectrum import EIS_Spectrum


class Instrument(Enum):
    """
    Simple enumeration class to easily reference instrument types.

    Attributes
    ----------
    GAMRY
        GAMRY instrument producing .DTA files (value [str]: "GAMRY")
    BIOLOGIC
        BIOLOGIC instrument producing .mpt files (value [str]: "BIOLOGIC")
    """

    GAMRY = "GAMRY"
    BIOLOGIC = "BIOLOGIC"


class Experiment:
    """
    Simple class object to load and parse the EIS experimental datafiles produced by various
    instrument brands.

    Parameters
    ----------
    path: str
        path to the experimental file to parse
    
    Raises
    ------
    FileNotFound
        exception raised when the provided path do not represent a vaid file path
    UnknownFileExtension
        exception raised when the file extension does not match the supported ones
    """

    def __init__(self, path: str) -> None:

        if not isfile(path):
            raise FileNotFound(path)

        self.__path = path
        self.__instrument = None
        self.__frequency = None
        self.__impedance = None

        extension = path.split(".")[-1]
        if extension == "DTA":
            self.__instrument = Instrument.GAMRY
            self.__frequency, self.__impedance = readGamry(self.__path)
        
        elif extension == "mpt":
            self.__instrument = Instrument.BIOLOGIC
            self.__frequency, self.__impedance = readBioLogic(self.__path)

        else:
            raise UnknownFileExtension(extension)
    
    @property
    def instrument(self) -> Instrument:
        """
        Brand/model of the instrument used to collect the experimental data

        Returns
        -------
        Instrument
            the name of the instrument
        """
        return self.__instrument
    
    @property
    def spectrum(self) -> EIS_Spectrum:
        """
        Experimental EIS spectrum

        Returns
        -------
        EIS_Spectrum
            an EIS_Spectrum object containing the experimental data collected during the
            experiment
        """
        return EIS_Spectrum(self.__frequency, self.__impedance)