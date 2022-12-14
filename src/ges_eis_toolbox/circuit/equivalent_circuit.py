from __future__ import annotations

import warnings, numpy
from impedance.models.circuits import CustomCircuit
from typing import Dict, List, Union

from ges_eis_toolbox.utils import remove_numbers
from ges_eis_toolbox.exceptions import InvalidParametrization
from ges_eis_toolbox.circuit.circuit_string import CircuitString


class EquivalentCircuit:
    """
    Class to hold the data relative to a parametrized EIS equivalent circuit. The class wraps
    the simulation routine provided by the impedance.py package. The class provides __getitem__
    and __setitem__ methods to access/set the component values by name. The __eq__ and __ne__
    methods are also provided to check strict equality (equality between electrically equivalent
    circuits will evaluate to true only if the circuit topology and component numbering matches)

    Parameters
    ----------
    circuit: Union[str, CircuitString]
        string representation of a circuit either in form of a CircuitString object or in the
        form of a valid string (used to automatically inizialize a CircuitString object)
    parameters: Union[None, Dict[str, float]]
        a dictionary containing the list of components (as keys) associated to their value.
        If set to None will inizialize all the components values to None

    Raises
    ------
    TypeError
        Exception raised if the circuit parameter type does no match the required one
    """

    def __init__(
        self,
        circuit: Union[str, CircuitString],
        parameters: Union[None, Dict[str, float]] = None,
    ) -> None:

        if type(circuit) != str and type(circuit) != CircuitString:
            raise TypeError

        self.__circuit = CircuitString(circuit) if type(circuit) == str else circuit
        self.__circuit._validate()

        self.__parameters = (
            parameters
            if parameters
            else {key: None for key in self.__circuit.list_components()}
        )
        self.__validate()

    def __getitem__(self, name: str) -> float:
        if name in self.__parameters:
            return self.__parameters[name]
        else:
            raise ValueError

    def __setitem__(self, name: str, value: float) -> float:
        if name in self.__parameters:
            self.__parameters[name] = value
        else:
            raise ValueError
    
    def __eq__(self, other: EquivalentCircuit) -> bool:

        if str(self.circuit_string) != str(other.circuit_string):
            return False
        elif self.parameters != other.parameters:
            return False
        else:
            return True
    
    def __ne__(self, other: EquivalentCircuit) -> bool:
        return not (self == other)

    def __validate(self, check_none: bool = False) -> None:
        """
        Validate the parameter set by checking the matching between the circuit string and
        the list of components.

        Raises
        ------
        InvalidParametrization:
            exception raised when a mismatch between the circuit structure and the parameter list
            is detected
        ValueError:
            exception raised (if check_none is set to True) when a None value is encountered
            in the parameter list
        """
        string_components = self.__circuit.list_components()
        string_components.sort()

        params_components = [k for k in self.__parameters.keys()]
        params_components.sort()

        if len(string_components) == len(params_components):
            for name_1, name_2 in zip(string_components, params_components):
                if name_1 != name_2:
                    raise InvalidParametrization("Mismatch between component naming.")
        else:
            msg = "Mismatch between the number of components and parameters."
            raise InvalidParametrization(msg)

        if check_none:
            for value in self.__parameters.values():
                if value is None:
                    raise ValueError

    def reorder(self) -> None:

        self.__circuit, ct = self.__circuit.reorder().reorder_labels()
        self.__parameters = {ct[key]: value for key, value in self.__parameters.items()}
        self.__validate()

    def simulate(self, frequency: List[float]) -> numpy.ndarray:
        """
        Simulate the circuit over the range of specified frequencies using the set of user-defined
        parameters.

        Parameters
        ----------
        frequency: List[float]
            list of frequency point on which the impedance must be computed

        Returns
        -------
        numpy.ndarray
            the array of complex impedance values computed in the point specified
        """

        self.__validate(check_none=True)

        f = numpy.array(frequency)
        
        constants = {}
        for key, parameter in self.__parameters.items():
            
            ctype = remove_numbers(key)
            cnumber = key.strip(ctype)
            symbol = f"{ctype}{cnumber}"

            if type(parameter) == list:
                for i, p in enumerate(parameter):
                    constants[f"{symbol}_{i}"] = float(p)
            else:
                constants[symbol] = float(parameter)

        circuit = CustomCircuit(self.__circuit.value, constants=constants)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            Z = circuit.predict(f, use_initial=True)

        return Z

    @property
    def circuit_string(self) -> CircuitString:
        """
        The CircuitString object representing the equivalent circuit
        """
        return self.__circuit
    
    @property
    def parameters(self) -> Dict[str, float]:
        """
        The parameters defining the equivalent circuit
        """
        return self.__parameters
