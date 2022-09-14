from __future__ import annotations
from typing import Dict, List, Tuple

from ges_eis_toolbox.utils import remove_numbers
from ges_eis_toolbox.exceptions import InvalidSyntax, InvalidComponent

_VALID_COMPONENTS = ("C", "CPE", "G", "Gs", "K", "L", "La", "R", "T", "TLMQ", "W", "Wo", "Ws")

class CircuitString:
    """
    Class to store the properties of the string representation of an EIS equivalent circuit
    according to the convention adopted in the impedance.py library. The class implements the
    basic manupulation functions to decompose, compare and analyze the circuit properties.
    The class implements a `__str__` and a `__repr__` methods returning the content of the string.
    The class also implements the `__add__` and `__iadd__` operations to concatenate other circuit
    string representations.

    Argument
    --------
    string: str
        the string representing the EIS equivalent circuit
    
    Raise
    -----
    TypeError
        exception raised when the input `string` is not of type `str`
    """

    def __init__(self, string: str) -> None:
        
        if type(string) != str:
            raise TypeError

        self.__string = string

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
            blocks.append(CircuitString(substing))

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

        # Replace double ",," symbols with ","
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
        return CircuitString(buffer)

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
            string = self.__string.lstrip("p(").rstrip(")")

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
        counter = {x : 0 for x in component_types}
        
        # Compile a conversion table by assigning a new name to each component
        conversion_table = {}
        for old_symbol in current_order:
            ctype = remove_numbers(old_symbol)          # Get the component type
            new_symbol = ctype + str(counter[ctype])    # Generate a new symbol for the component
            counter[ctype] += 1                         # Increment the counter
            conversion_table[old_symbol] = new_symbol   # Add the new correspondence to the conversion table
        
        # Define a new string in which the old component symbols are exchanged with the new ones.
        # Start by iterating from index (idx) 0 and define a buffer to store the part of string
        # that has been parsed. For each component iterate over the string until a stop symbol
        # [",", "-", "p", "(", ")"] is encountered. Once the component symbol has been obtained
        # swap it with the one in the comnversion table. Copy the stop symbol and move to the next
        # iteration.
        idx = 0
        new_string = ""
        while idx<len(self.__string):
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

    @property
    def value(self):
        """
        The circuit string (read-only).
        """
        return self.__string

    @property
    def number_of_components(self):
        """
        The total number of components in the circuit.
        """
        return len(self.list_components())

    @property
    def is_simple_series(self):
        """
        True if the string represent a circuit generated by the simple series
        combination of components (no element in parallel), else False
        """
        return False if any("p" in x.value for x in self.decompose_series()) else True
    
    @property
    def is_pure_parallel(self):
        """
        True if the string represent a single parallel circuit block, else False
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
    