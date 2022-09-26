from __future__ import annotations

import numpy
import numpy as np
from os import mkdir
from os.path import isdir, abspath
from dataclasses import dataclass
from math import factorial
from typing import Any, Dict, List, Union
from multiprocessing import Pool, cpu_count

from ges_eis_toolbox.circuit.circuit_string import CircuitString
from ges_eis_toolbox.circuit.equivalent_circuit import EquivalentCircuit
from ges_eis_toolbox.spectra.spectrum import EIS_Spectrum
from ges_eis_toolbox.database.data_entry import DataPoint, DataOrigin


class Range:
    """
    Simple multiparametric range class capable, given an index, of mapping one or more parameters
    in a range between a maximum and a minimum.

    Parameters
    ----------
    min: Union[float, List[float]]
        float or list of float defining the minimum of the range
    max: Union[float, List[float]]
        float or list of float defining the maximum of the range

    Raises
    ------
    TypeError
        exception raised when the argument type does not match the expected ones (float or List[float])
    ValueError
        exception raised when either min and max lists have different length or if at least
        one of the min values is greater than the correspondent max value
    """

    def __init__(
        self, min: Union[float, List[float]], max: Union[float, List[float]]
    ) -> None:

        self.__min = self.__validate(min)
        self.__max = self.__validate(max)

        if len(self.__min) != len(self.__max):
            raise ValueError("The 'min' and 'max' arguments must have the same length")

        for x, y in zip(self.__max, self.__min):
            if x < y:
                raise ValueError(
                    "The 'max' value must always be greater or equal to the 'min' one"
                )

        self.__nparams = len(self.__max)

    def __validate(self, input: Any) -> List[float]:
        """
        Validation function capable of raising an exception when the wrong parameter type is
        passed as an argument. The function returns the input values in the List[float] format.

        Parameter
        ---------
        input: Any
            input parameter to validate (either min or max)

        Raises
        ------
        TypeError
            exception raised when the argument type does not match the expected ones (float or List[float])

        Returns
        -------
        List[float]
            the input parameter in the form of List[float]
        """

        if type(input) != float and type(input) != list:
            raise TypeError(
                "The 'min' and 'max' arguments must be of type 'float' or 'List[float]'"
            )
        elif input == []:
            raise ValueError("The 'min' and 'max' arguments cannot be empty lists")
        elif type(input) == float:
            return [input]
        else:
            return [float(x) for x in input]

    def __len__(self) -> int:
        """
        Define the length of the object as the number of parameters contained in the range

        Returns
        -------
        int
            the number of parameters considered in the object
        """
        return self.__nparams

    @property
    def min(self) -> List[float]:
        """
        The minimum extreme of the range

        Returns
        -------
        List[float]
            The list of float values encoding the minimum extreme of the range
        """
        return self.__min

    @property
    def max(self) -> List[float]:
        """
        The maximum extreme of the range

        Returns
        -------
        List[float]
            The list of float values encoding the maximum extreme of the range
        """
        return self.__max

    @property
    def delta(self) -> List[float]:
        """
        The length of the range sides

        Returns
        -------
        List[float]
            The list of float values encoding the length of each side of the range
        """
        return [x - y for x, y in zip(self.__max, self.__min)]

    def generate_step(self, index: List[int], steps: int):
        """
        Given the homogeneous subdivision of the range in `steps` parts, comutes the values of
        grid associated the point described by the list of indeces provided as the `index` variable

        Parameters
        ----------
        index: List[int]
            the list encoding the coordinates of the desired point in the homogeneouly sampled
            space
        steps: int
            the number of steps in which each dimension of the range is subdivided

        Raises
        ------
        ValueError
            exception raised when either the `index` or the `steps` parameter assume an invalid value

        Returns
        -------
        List[float]
            the coordinate of the selected point on the grid
        """

        if steps <= 0:
            raise ValueError("The 'steps' parameter must be a positive integer")

        for i in index:
            if i < 0 or i >= steps:
                raise ValueError(
                    "The 'index' parameter must be a non-negative integer smaller than 'steps'"
                )

        return [
            min + delta * i / (steps - 1)
            for min, delta, i in zip(self.min, self.delta, index)
        ]


class Generator:
    """
    Simple genrator class capable of running multiple simulation of a given circuit exploring,
    with an omogeneus scheme, a defined range of component parameters. The class take advantage
    of the series permutation symmetry of the circuit to reduce the number of simulation to
    be executed.

    Parameters
    ----------
    circuit: CircuitString
        the circuit string object representing the base circuit to simulate
    ranges: Dict[str, Dict[str, Range]]
        the dictionary encoding the range of component values to be explored. The key of the
        dictionary represents the permutation base groups encoding the circuit while the values
        encode the range of values associated to each component. These are in turn organized in
        a dictionary in which the key are represented by the component symbols while the values
        are represented by Range objects.
    simulation_limit: int
        sets the maximum number of simulations to be executed by the code (default: 1e+6). Please
        notice how this is different from the real number of simulations that is instead computed
        based on the circuit structure.
    steps_limit: Union[int, None]
        sets the maximum number of steps for each degree of freedom to be explored in the
        simulation. If set to None, will leave the step number free to approach the value that
        better fits the `simulation_limit` parameter.

    Raises
    ------
    TypeError
        exception raised if the `circuit` parameter is not of type `CircuitString`
    ValueError
        exception raised if a mismatch exists between the key of the ranges parameters and the
        one generated by the `permutation_base_groups` function of the `CircuitString` class
    """

    def __init__(
        self,
        circuit: CircuitString,
        ranges: Dict[str, Dict[str, Range]],
        simulation_limit: int = 1000000,
        steps_limit: Union[int, None] = None,
    ) -> None:

        if type(circuit) != CircuitString:
            raise TypeError("The 'circuit' parameter must be of type 'CircuitString'.")

        self.__circuit = circuit
        self.__groups = circuit.permutation_base_groups()

        if sorted(self.__groups.keys()) != sorted(ranges.keys()):
            raise ValueError("Key mismatch between circuit base groups and 'ranges'.")

        self.__ranges = ranges

        steps = 1
        while True:

            if steps_limit is not None:
                if steps + 1 > steps_limit:
                    break

            if self.__compute_number_of_simulations(steps + 1) < simulation_limit:
                steps += 1
            else:
                break

        self.__steps = steps
        self.__number_of_simulations = self.__compute_number_of_simulations(self.__steps)

    def __compute_number_of_simulations(self, steps: int) -> int:
        """
        Computes the number of simulations resulting from an homogeneous subdivision of the
        circuit configuration space in a given number of steps. The function takes into account
        the reduction in the number of simulations associated with the series permutation symmetry
        of the circuit.

        Parameters
        ----------
        steps: int
            number of steps in which each dimension of the configuration space is divided

        Raises
        ------
        ValueError
            exception raised if the step parameter is not a positive integer

        Returns
        -------
        int
            the number of simulations associated to a given `steps` value
        """
        # Check the validity of the given step value
        if type(steps) != int or steps <= 0:
            raise ValueError("The 'step' parameter must be a positive integer")

        simulations = 1
        for block, conversion_tables in self.__groups.items():

            # Get the number of equivalent elements for each symmetry block type
            number_of_elements = len(conversion_tables)

            # Compute the total number of variables for each block
            number_of_variables = 0
            for r in self.__ranges[block].values():
                number_of_variables += len(r)

            # If there is only one element for this block type there is no symmetry advantage
            # so just treat the variable as an independent degree of freedom
            if number_of_elements == 1:
                simulations *= steps**number_of_variables

            # If there are more than one element for this type of block take into account the
            # advantage derived from the symmetry and apply to each variable type (x) the relation
            # x1 >= x2 >= x3 .... >= xM
            else:
                composite_steps = self.__number_of_configurations(number_of_elements, steps)
                simulations *= composite_steps**number_of_variables

        return simulations

    @staticmethod
    def __number_of_configurations(elements: int, steps: int) -> int:
        """
        Compute the number of configurations generated from a given number of equivalent
        variable `elements` mapped with a given number of `steps`. If `K` represents the
        number of `elements` (x1, x2, ..., xK) and `N` the number of steps, the function
        returns the value `(N+K-1)!/(K!*(N-1)!) representing all the configurations responding
        to x1 >= x2 >= ... >= xK.

        Parameters
        ----------
        elements: int
            the number of equivalent elements in the circuit blocks
        steps: int
            the number of steps in which a given variable type is subdivided

        Raises
        ------
        ValueError
            if either the number of steps of elements assumes negative value

        Returns
        -------
        int
            the number of permutationally invariant configurations of a variable
        """

        if elements < 0 or steps < 0:
            raise ValueError("Both 'elements' and 'steps' must be non negative integers")

        return int(
            factorial(steps + elements - 1) / (factorial(elements) * factorial(steps - 1))
        )

    def generate_parameterization(self, index: int) -> Dict[str, Union[float, List[float]]]:
        """
        Generate the parameterization associated to a given index taking into account the series
        permutation symmetry of the circuit elements.

        Parameters
        ----------
        index: int
            the index of the desired circuit configuration

        Raises
        ------
        ValueError
            exception raised if the given index is invalid (either negative of greated then
            the number of simulations)

        Returns
        -------
        Dict[str, Union[float, List[float]]]
            The dictionary encoding the parameterization of the circuit. The key of the dictionary
            encode the component symbol while the values the associated parameters. If the only one
            parameter is associated to a given component the value will be of type float else of
            type List[float].
        """

        # Check that the given index assumes a valid value
        if index < 0 or index >= self.__number_of_simulations:
            raise ValueError(
                "The index value must be a non-negative integer lower than the maximum number of simulations"
            )

        pivot = index
        parameter_list: Dict[str, List[float]] = {}

        # Iterate over all permutation base groups exctracting the base block string and the
        # conversion tables required to generate the correspondence with the real circuit
        for block, conversion_tables in self.__groups.items():

            # Get a list of components in the symmetry base block
            base_components = CircuitString(block).list_components()

            # Get the number of instances of the current block in the real circuit and compute
            # the number of steps required to explore, for each independent parameter, the whole
            # configuration space of circuit parameters
            number_of_elements = len(conversion_tables)
            cumulative_steps = self.__number_of_configurations(
                number_of_elements, self.__steps
            )

            # Iterate over each component of the base block
            for base_component in base_components:

                # Get the number of parameters associated to each component based on the
                # user supplied range. These represent the number of independent variables to
                # scan for each block
                number_of_parameters = len(self.__ranges[block][base_component])

                # Define a buffer variable to hold the step of each parameter associated to a
                # given component (required because by iterating over the parameter varable
                # before iterating on the number of instances will generate partial component
                # parameterizations)
                buffer: Dict[str, List[int]] = {}

                # Iterate over each independent parameter
                for pidx in range(number_of_parameters):

                    # Compute the index that defines a (multi-element) variable block
                    parameter_index = pivot % cumulative_steps

                    # Update the pivot variable to compute the index quotient to be passed to
                    # the next parameter-based iterations
                    pivot = int((pivot - parameter_index) / cumulative_steps)

                    # Define an order list that, for the current independent variable. It will
                    # hold the order of the step associated to each permutationally invariant
                    # element. Iterate over each element to fill the list.
                    order_list = []
                    for eidx in range(number_of_elements):

                        # Define a 'total_configurations' variable to hold the sum of the number of
                        # configurations of the sub-group of elements explored when reaching
                        # the current 'order' value.
                        total_configurations, order = 0, 0
                        while True:

                            # Compute the number of configurations existing in a sub-group of
                            # elements of one dimension lower than the current one mapped with
                            # a number of steps not greater than the currently explored order
                            new_configurations = self.__number_of_configurations(
                                number_of_elements - eidx - 1, order + 1
                            )

                            # If the total number of configurations exceed the computed index,
                            # the lastly considered order is too high. If True stop the
                            # iteration, else increment the 'total_configurations' conuter
                            # and examine the next order
                            if total_configurations + new_configurations > parameter_index:
                                break
                            else:
                                total_configurations += new_configurations
                                order += 1

                        # Append the current order to the list
                        order_list.append(order)

                        # Update the parameter_index variable to prepare the next iteration.
                        # To do so, subtract the number of configurations from the parameter
                        # index.
                        parameter_index -= total_configurations

                    # Iterate over the list of conversion tables to evaluate the value of the
                    # current parameter type associad to the circuit component belonging to
                    # the current base group
                    for eidx, table in enumerate(conversion_tables):

                        # Compute the inverse conversion table to convert the label of the base block
                        # to the corresponding symbol in the real circuit
                        inv_table = {y: x for x, y in table.items()}
                        symbol = inv_table[base_component]

                        if pidx == 0:
                            buffer[symbol] = [order_list[eidx]]
                        else:
                            buffer[symbol].append(order_list[eidx])

                # Use the buffer to update the parameter_list with the appropriate step
                for key, value in buffer.items():

                    if key in parameter_list:
                        raise RuntimeError(
                            f"Unexpected multiple entry encountered for component symbol '{key}'"
                        )

                    component_range = self.__ranges[block][base_component]
                    parameter_list[key] = component_range.generate_step(value, self.__steps)

        return {
            key: val if len(val) != 1 else val[0] for key, val in parameter_list.items()
        }

    def run(
        self,
        frequency: Union[List[float], numpy.ndarray],
        cores: int = -1,
        basename: str = "simulation",
        folder: str = ".",
    ) -> None:
        """
        Run all the simulations in parallel using a static scheduling and dump the results to
        individual simulation files.

        Parameters
        ----------
        frequency: Union[List[float], numpy.ndarray]
            the list (or numpy array) containing the float frequency values at which the
            circuit must be simulated
        cores: int
            the number of cores to be used by the simulation process. If set to `-1` (default)
            will use all the cores available on the machine
        basename: str
            the basename of the generated datafiles (default: `simulation`). e.g. if basename
            is set to `sim` the files generated will be named `sim_X.json` where `X` represents
            the simulation index.
        folder: str
            the path to the folder in which the datafiles must be saved. (default: ".")

        Raises
        ------
        ValueError
            exception raised if the number of cores selected by the user is invalid
        """
        # Check the user provided number of cores
        if cores == -1:
            cores = cpu_count()
        elif cores <= 0:
            raise ValueError("Invalid number of cores selected.")

        # Check if the folder indicated by the user exists, else create it
        if isdir(folder) == False:
            mkdir(folder)
        folder = abspath(folder)

        # Set up an equal division of the number of jobs to be assigned to each process
        surplus = self.__number_of_simulations % cores
        steps = int((self.__number_of_simulations - surplus) / cores)
        start_points = [(i + 1) * steps for i in range(cores)]
        start_points[-1] += surplus

        # Generate the list of tasks to be performed by each process
        tasks = []
        for core in range(cores):
            start = 0 if core == 0 else start_points[core - 1]
            end = start_points[core]
            freq = frequency if type(frequency) == list else [f for f in frequency]
            tasks.append(Task(self, start, end, freq, basename, folder))

        # Run all the processes in parallel
        with Pool(processes=cores) as pool:
            pool.map(job_engine, tasks)

    @property
    def number_of_simulations(self) -> int:
        """
        The number of simulation that will be executed

        Returns
        -------
        int
            the total number of sumulations
        """
        return self.__number_of_simulations

    @property
    def number_of_steps(self) -> int:
        """
        The number of steps in which each parameter will be samples

        Returns
        -------
        int
            the total number of steps for each variable
        """
        return self.__steps

    @property
    def circuit(self) -> CircuitString:
        """
        The circuit string representing the circuit to be simulated

        Returns
        -------
        CircuitString
            the `CrcuitString` object representing the circuit to be simulated
        """
        return self.__circuit


@dataclass
class Task:
    """
    Simple dataclass to hold the parameters required to run a simulation with the job_engine
    function.

    Parameters
    ----------
    gen: Generator
        the object capable of generating, starting from a given index, the circuit parameterization
        to be considered in the simulation.
    start: int
        the starting value of the range in which the index must be sampled (included in the simulation)
    end: int
        the end value of the range in which the index must be sampled (excluded from the simulation)
    frequency: numpy.ndarray
        the list holding the frequency values at which the circuit must be simulated
    basename: str
        the basename of the generated datafiles. e.g. if basename is set to `sim` the files
        generated will be named `sim_X.json` where `X` represents the simulation index.
    folder: str
        the path to the folder in which the datafiles must be saved.
    """

    gen: Generator
    start: int
    end: int
    frequency: List[float]
    basename: str
    folder: str


def job_engine(task: Task):
    """
    Runs all the simulations encoded by the selected simulation task and saves the data into
    a file.

    Parameters
    ----------
    task: Task
        the dataclass encoding the simulation parameters
    """
    for i in range(task.end - task.start):
        idx = task.start + i
        params = task.gen.generate_parameterization(idx)
        circuit = EquivalentCircuit(task.gen.circuit, parameters=params)
        Z = circuit.simulate(task.frequency)
        spectrum = EIS_Spectrum(np.array(task.frequency), Z)
        dp = DataPoint(DataOrigin.Real, equivalent_circuit=circuit, spectrum=spectrum)
        dp.save(f"{task.basename}_{idx}", folder=task.folder)
