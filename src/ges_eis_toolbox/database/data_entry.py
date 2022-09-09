from __future__ import annotations

import json
from os.path import isfile
from dataclasses import dataclass
from enum import Enum

from ges_eis_toolbox.exceptions import FileNotFound
from ges_eis_toolbox.circuit.equivalent_circuit import EquivalentCircuit
from ges_eis_toolbox.spectra.spectrum import EIS_Spectrum


class DataOrigin(Enum):
    Real = "REAL"
    Simulation_noiseless = "NOISELESS_SIMULATION"


@dataclass
class DataPoint:

    origin: DataOrigin
    user: str = None
    eq_circuit: EquivalentCircuit = None
    spectrum: EIS_Spectrum = None

    def save(self, filename: str) -> None:
        content = {}
        content["origin"] = self.origin.value
        content["user"] = self.user
        content["circuit"] = self.eq_circuit.circuit_string.value
        content["parameters"] = self.eq_circuit.parameters
        content["frequency"] = self.spectrum.frequency
        content["real Z"] = [z.real for z in self.spectrum.impedance]
        content["imag Z"] = [z.imag for z in self.spectrum.impedance]

        with open(filename, "w") as file:
            file.write(json.dumps(content, indent=4))

    @classmethod
    def load(cls, path: str) -> DataPoint:

        if not isfile(path):
            raise FileNotFound(path)

        with open(path, "r") as file:
            data = json.load(file)

        cls.origin = DataOrigin(data["origin"])
        cls.user = data["user"]

        cls.eq_circuit = EquivalentCircuit(
            data["circuit"],
            parameters=data["parameters"],
        )

        impedance = [real + 1j * imag for real, imag in zip(data["real Z"], data["imag Z"])]
        cls.spectrum = EIS_Spectrum(data["frequency"], impedance)

        return cls
