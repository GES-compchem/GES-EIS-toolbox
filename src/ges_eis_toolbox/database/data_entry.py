from __future__ import annotations

import json
import matplotlib.pyplot as plt

from os.path import isfile, join, abspath
from dataclasses import dataclass
from enum import Enum
from typing import Union
from impedance.visualization import plot_bode, plot_nyquist

from ges_eis_toolbox.exceptions import FileNotFound
from ges_eis_toolbox.circuit.equivalent_circuit import EquivalentCircuit
from ges_eis_toolbox.spectra.spectrum import EIS_Spectrum
from ges_eis_toolbox.database.experiment import Instrument


class DataOrigin(Enum):
    Real = "REAL"
    Simulation_noiseless = "NOISELESS_SIMULATION"


@dataclass
class DataPoint:
    """
    Datacass holding all the data to be stored in the database. The class can be used to
    record both experimental and simulation data. The class implements a save and a
    load methods to save and load JSON formatted datafiles.

    Arguments
    ---------
    origin: DataOrigin
        the origin of the data contained in the object
    user: Union[str, None]
        the name of the user that performed the fitting of the experimental data, None in the
        case of a simulation
    instrument: Union[Instrument, None]
        the instrument used in recording the EIS measurement, None in the case of a simulation
    equivalent_circuit: EquivalentCircuit
        the equivalent circuit used in the fit/simulation
    spectrum: EIS_Spectrum
        the data relative to the EIS spectrum recorded/simulated
    """

    origin: DataOrigin
    user: Union[str, None] = None
    instrument: Union[Instrument, None] = None
    equivalent_circuit: EquivalentCircuit = None
    spectrum: EIS_Spectrum = None

    def save(self, name: str, folder: str = ".") -> None:
        """
        Saves all the data to a JSON formatted file

        Parameters
        ----------
        name: str
            the name of the destination file without extension. All the trailing fields
            separated by `.` will be stripped from the final name
        folder: str
            the folder path to which the file must be saved (default: .)
        """
        content = {}
        content["origin"] = self.origin.value
        content["user"] = self.user
        content["instrument"] = None if self.instrument is None else self.instrument.value
        content["circuit"] = self.equivalent_circuit.circuit_string.value
        content["parameters"] = self.equivalent_circuit.parameters
        content["frequency"] = [f for f in self.spectrum.frequency]
        content["real Z"] = [z.real for z in self.spectrum.impedance]
        content["imag Z"] = [z.imag for z in self.spectrum.impedance]

        path = join(abspath(folder), name.split(".")[0] + ".json")
        with open(path, "w") as file:
            file.write(json.dumps(content, indent=4))

    @classmethod
    def load(cls, path: str) -> DataPoint:
        """
        Returns a DataPoint object from a JSON formatted file located at the specified path

        Parameters
        ----------
        path: str
            the path to the .json file containing the DataPoint information

        Raises
        ------
        FileNotFound
            exception raised is the specified path is not valid
        """

        if not isfile(path):
            raise FileNotFound(path)

        with open(path, "r") as file:
            data = json.load(file)

        origin = DataOrigin(data["origin"])
        user = data["user"]
        instrument = Instrument(data["instrument"]) if data["instrument"] else None

        equivalent_circuit = EquivalentCircuit(
            data["circuit"],
            parameters=data["parameters"],
        )

        impedance = [real + 1j * imag for real, imag in zip(data["real Z"], data["imag Z"])]
        spectrum = EIS_Spectrum(data["frequency"], impedance)

        return cls(origin, user, instrument, equivalent_circuit, spectrum)

    def bode_plot(self, ax) -> None:

        freq = self.spectrum.frequency
        Z_exp = self.spectrum.impedance
        Z_sim = self.equivalent_circuit.simulate(freq)

        plot_bode(ax, freq, Z_exp, fmt="o")
        plot_bode(ax, freq, Z_sim, fmt="-")
        plt.legend(["Data", "Fit"])

    def nyquist_plot(self, ax) -> None:

        freq = self.spectrum.frequency
        Z_exp = self.spectrum.impedance
        Z_sim = self.equivalent_circuit.simulate(freq)

        plot_nyquist(ax, Z_exp, fmt="o")
        plot_nyquist(ax, Z_sim, fmt="-")
        plt.legend(["Data", "Fit"])
