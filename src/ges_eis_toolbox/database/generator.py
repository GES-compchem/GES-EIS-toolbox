from math import factorial
from typing import Any, Dict, List, Union
from ges_eis_toolbox.circuit.circuit_string import CircuitString


class Range:
    """
    Simple multiparametric range class capable of generating sequence of values given a maximum
    and a minimum.

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
    Simple genrator class capable of running multiple simulation of a given circuit exploring
    with an omogeneus scheme a defined range of component parameters. The class take advantage
    of the series permutation symmetry of the circuit to reduce the number of simulation to
    be executed.

    Parameters
    ----------
    circuit: CircuitString
        the circuit string object representing the base circuit to simulate
    ranges: Dict[str, Dict[str, Range]]
        the dictionary encoding the range of component values to be explored. The key of the
        dictionary represent the permutation base groups encoding the circuit while the values
        encode the range of values associated to each component. These are in turn organized in
        a dictionary in which the key are represented by the component symbols while the values
        are represented by Range objects.
    simulation_limit: int
        sets the maximum number of simulation to be executed by the code (default: 1e+6). Please
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

        if elements < 0 or steps < 0:
            raise ValueError("Both 'elements' and 'steps' must be non negative integers")

        return int(
            factorial(steps + elements - 1) / (factorial(elements) * factorial(steps - 1))
        )

    def generate_parameterization(self, index: int) -> Dict[str, List[float]]:

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

            # Get a list of components in the base block
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

                    # Update to the pivot variable
                    pivot = int((pivot - parameter_index) / cumulative_steps)

                    # Define an order list that, for the current independent variable, will
                    # hold the order of the step associated to each permutationally invariant
                    # element. Iterate over each element to fill the list.
                    order_list = []

                    for eidx in range(number_of_elements):

                        # Define a 'total_configurations' variable to hold the sum of the number of
                        # configurations of the sub-group of elements explored when reaching
                        # the current 'order' value
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

        return parameter_list

    @property
    def number_of_simulations(self) -> int:
        return self.__number_of_simulations

    @property
    def number_of_steps(self) -> int:
        return self.__steps
