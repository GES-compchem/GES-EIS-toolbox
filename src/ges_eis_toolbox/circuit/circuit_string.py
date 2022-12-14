from __future__ import annotations
from typing import Dict, List, Tuple

from ges_eis_toolbox.utils import remove_numbers
from ges_eis_toolbox.exceptions import InvalidSyntax, InvalidComponent

_VALID_COMPONENTS = ("C", "CPE", "G", "Gs", "K", "L", "La", "R", "T", "TLMQ", "W", "Wo", "Ws")

class CircuitString:
    """
    Class to store the properties of the string representation of an EIS equivalent circuit
    according to the convention adopted in the `impedance.py` library. The class implements the
    basic manupulation functions to decompose, compare and analyze the circuit properties.
    The class implements a `__str__` and a `__repr__` methods returning the content of the string.
    The class also implements the `__add__` and `__iadd__` operations to concatenate other circuit
    string representations.

    Arguments
    ---------
    string: str
        the string representing the EIS equivalent circuit
    validate: bool
        if True (default) will automatically run a validation of the syntax of the input string

    Raise
    -----
    TypeError
        exception raised when the input `string` is not of type `str`
    """

    def __init__(self, string: str, validate: bool = True) -> None:

        if type(string) != str:
            raise TypeError

        self.__string = string

        if validate:
            self._validate()

    def __str__(self) -> str:
        return self.__string

    def __repr__(self) -> str:
        return self.__string

    def __add__(self, x: CircuitString) -> CircuitString:
        string = "-".join([self.__string, x.value])
        return CircuitString(string)

    def __iadd__(self, x: CircuitString) -> CircuitString:
        self.__string = "-".join([self.__string, x.value])
        return self

    def _validate(self) -> None:
        """
        Validate the circuit string by checking the type of components used and the formatting

        Raises
        ------
        IvalidSyntax:
            exception raised when an invalid circuit string syntax is used
        InvalidComponent:
            exception raised when an invalid component is used in the circuit definition
        """
        if " " in self.__string:
            raise InvalidSyntax("Circuit sting should not contain spaces")

        if self.__string.count("(") != self.__string.count(")"):
            raise InvalidSyntax("Mismatch in the number of open and closed brackets")

        if self.__string.count("(") != self.__string.count("p"):
            raise InvalidSyntax("Mismatch between parallel units and number of brackets")

        component_types = set(self.remove_numbers().list_components())
        for ctype in component_types:
            if ctype not in _VALID_COMPONENTS:
                raise InvalidComponent(ctype)

    def decompose_series(self) -> List[CircuitString]:
        """
        Returns, as a list of `CircuitString` objects, the decomposition of the circuit string
        in terms of sub-circuits that, when connected in series, represent the starting circuit.

        Returns
        -------
        List[CircuitString]
            a list containing the various sub-circuits strings
        """

        # Search for the break points dividing series sub-circuit blocks (the parallel circuit
        # elements are left untouched as a p(...) sub-circuit string)
        counter = 0
        break_points = []
        for i, char in enumerate(self.__string):
            if char == "-" and counter == 0:
                break_points.append(i)
            elif char == "(":
                counter += 1
            elif char == ")":
                counter -= 1
        break_points.append(len(self.__string))

        # Break the string along the obtained breakpoints and generate the wanted list of
        # sub-circuit strings
        blocks = []
        for i, end in enumerate(break_points):
            start = 0 if i == 0 else break_points[i - 1]
            substing = self.__string[start:end].strip("-")
            blocks.append(CircuitString(substing, validate=False))

        return blocks

    def list_components(self, sort: bool = True) -> List[str]:
        """
        Returns the labels associated to each component in the circuit in alphabetical order

        Returns
        -------
        List[str]
            the list of the label associated to each component in the circuit
        """
        # remove the parallel circuit marker
        buffer = self.__string.replace("p", "")

        # replace all the blocks separators with ","
        for char in ["(", ")", "-"]:
            buffer = buffer.replace(char, ",")

        # Replace multiple "," symbols with "," by iterating until ",," cannot be found
        while buffer.count(",,") != 0:
            buffer = buffer.replace(",,", ",")

        # If the buffer starts or ends with "," remove the first/last character
        if buffer.startswith(","):
            buffer = buffer[1::]

        if buffer.endswith(","):
            buffer = buffer[0:-1]

        # split, sort and return the buffer
        buffer = buffer.split(",")

        if sort:
            buffer.sort()

        return buffer

    def remove_numbers(self) -> CircuitString:
        """
        Returns the circuit string without the component numbers (only symbols and component
        connectivity)

        Returns
        -------
        CircuitString
            the circuit string without the component numbers
        """
        buffer = remove_numbers(self.__string)
        return CircuitString(buffer, validate=False)

    def reorder(self) -> CircuitString:
        """
        Recursively orders the circuit string in order of sub-circuit complexity.

        Returns
        -------
        CircuitString
            the ordered circuit string
        """

        # If the circuit block is a purely parallel unit strip the external parallel block
        # identifiers ("p(" and ")"). Substitute the commas dividing the first layer of
        # components with "@" and divide the parallel branches and apply recursive reordering
        # join back the ordered parallel unit in a circuit string.
        if self.is_pure_parallel:
            string = self.__string[2:-1]

            level = 0
            buffer = ""
            for char in string:
                buffer += "@" if char == "," and level == 0 else char
                if char == "(":
                    level += 1
                elif char == ")":
                    level -= 1

            branches = [CircuitString(s) for s in buffer.split("@")]
            branches = [branch.reorder().value for branch in branches]
            branches.sort()
            new_string = "p(" + ",".join(branches) + ")"
            return CircuitString(new_string)

        # If the circuit block is a simple series of components (no parallel blocks) apply a
        # simple decomposition, sorting and repacking as CircuitString
        elif self.is_simple_series:
            components = [x.value for x in self.decompose_series()]
            components.sort()
            new_string = "-".join(components)
            return CircuitString(new_string)

        # If the circuit is more complex decompose in series sub-circuits and recursively call
        # the reoreder function. With the resultin pieces sort the list by alphabetical order
        # and rebuild the CircuitString by summation.
        else:
            blocks = self.decompose_series()
            blocks = [b.reorder().value for b in blocks]
            blocks.sort()
            obj = CircuitString(blocks[0])
            for block in blocks[1::]:
                obj += CircuitString(block)
            return obj

    def reorder_labels(self) -> Tuple[CircuitString, Dict[str, str]]:
        """
        Returns the circuit string in which the components labels have been assigned in
        progressive order, without gaps, to number each component type

        Returns
        -------
        CircuitString
            the circuit string with the updated labels
        Dict[str, str]
            dictionary mapping the old component symbol (key) to the new one
        """
        # Obtain the list of components in the same order in which they appear in the string
        current_order = self.list_components(sort=False)

        # Extract the component types from the component list and prepare a counter for each
        # of them to keep track of the label to be used
        component_types = set(self.remove_numbers().list_components())
        counter = {x: 0 for x in component_types}

        # Compile a conversion table by assigning a new name to each component
        conversion_table = {}
        for old_symbol in current_order:
            ctype = remove_numbers(old_symbol)  # Get the component type
            new_symbol = ctype + str(
                counter[ctype]
            )  # Generate a new symbol for the component
            counter[ctype] += 1  # Increment the counter
            conversion_table[
                old_symbol
            ] = new_symbol  # Add the new correspondence to the conversion table

        # Define a new string in which the old component symbols are exchanged with the new ones.
        # Start by iterating from index (idx) 0 and define a buffer to store the part of string
        # that has been parsed. For each component iterate over the string until a stop symbol
        # [",", "-", "p", "(", ")"] is encountered. Once the component symbol has been obtained
        # swap it with the one in the comnversion table. Copy the stop symbol and move to the next
        # iteration.
        idx = 0
        new_string = ""
        while idx < len(self.__string):
            buffer = ""
            for char in self.__string[idx::]:
                idx += 1
                if char in [",", "-", "p", "(", ")"]:
                    if buffer != "":
                        new_string += conversion_table[buffer]
                    new_string += char
                    break
                else:
                    buffer += char
            else:
                if buffer != "":
                    new_string += conversion_table[buffer]

        return CircuitString(new_string), conversion_table

    def permutation_base_groups(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Compute the decomposition of the circuit string in a set of permutation base groups.
        Each base group represents the base circuit structure associated to one or more series
        circuit blocks equivalent under component permutation and re-labelling.

        Returns
        -------
        Dict[str, List[Dict[str, str]]]
            the dictionary containing the data encoding each base group. The keys of the 
            dictionary encode the structure of the base circuit while the dictionary values
            represent a list, of lenght equal to the number of equivalent circuit blocks, containing
            the conversion tables encoding the correspondence between the components in the
            real circuit and the one in the base one.
        """

        # Reorder the circuit string to ensure that all the blocks are sorted according
        # to their string representation (similar blocks will be adjacent)
        ordered_circuit, conversion_table = self.reorder().reorder_labels()

        # Compute the inverse conversion table by exchanging key and values in the dictionary
        inv_conversion_table = {y: x for x, y in conversion_table.items()}

        # Decompose the ordered circuit sting in series blocks
        ordered_blocks = ordered_circuit.decompose_series()

        # Reorder the labels of each block in order to avoid differences between blocks derived
        # from differences in component numbering. Convert each block in its string representation
        blocks = [str(obj.reorder_labels()[0]) for obj in ordered_blocks]

        # Iterate over the list of all blocks and group/count the number of each equal adjacent
        # block. Save the conversion table of each block (referred to the reordered one) in a list.
        idx = 0
        unique_blocks = {}
        while idx < len(blocks):
            component = blocks[idx]
            if component in unique_blocks:
                raise RuntimeError("Undefined behaviour, please contact the developers")
            ctables = []
            while component == blocks[idx]:
                _, ct = ordered_blocks[idx].reorder_labels()
                ctables.append(ct)
                idx += 1
                if idx >= len(blocks):
                    break
            unique_blocks[component] = ctables
        
        # Use the initial conversion table to update all the conversion tables of the blocks
        buffer = {}
        for block, tables in unique_blocks.items():
            updated_tables = []
            for table in tables:
                ct = {inv_conversion_table[key] : value for key, value in table.items()}
                updated_tables.append(ct)
            buffer[block] = updated_tables

        return buffer

    @property
    def value(self) -> str:
        """
        The circuit string (read-only).

        Returns
        -------
        str
            the string representing the circuit
        """
        return self.__string

    @property
    def number_of_components(self) -> int:
        """
        The total number of components in the circuit.

        Returns
        -------
        int
            the total number of components in the string.
        """
        return len(self.list_components())

    @property
    def is_simple_series(self) -> bool:
        """
        True if the string represents a circuit generated by the simple series
        combination of components (no element in parallel), else False.

        Returns
        -------
        bool
            boolean representing if the string represents a circuit generated by the simple series
            combination of components.
        """
        return False if any("p" in x.value for x in self.decompose_series()) else True

    @property
    def is_pure_parallel(self) -> bool:
        """
        True if the string represents a single parallel circuit block, else False.

        Returns
        -------
        bool
            boolean representing if the string represents a single parallel circuit block.
        """
        if self.__string.startswith("p"):
            level = 0
            for char in self.__string[1::]:
                if char == "(":
                    level += 1
                elif char == ")":
                    level -= 1
                else:
                    if level == 0:
                        return False
            return True
        else:
            return False

    @property
    def is_ordered(self) -> bool:
        """
        True if the circuit string is ordered both in terms of series blocks and component
        labels, else false.

        Returns
        -------
        bool
            boolean representing if the circuit string is ordered.
        """
        ordered_string, _ = self.reorder().reorder_labels()
        return True if ordered_string.value == self.__string else False
