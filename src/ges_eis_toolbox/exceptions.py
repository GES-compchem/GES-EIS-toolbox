from ges_eis_toolbox.utils import remove_numbers

class FileNotFound(Exception):
    """
    Exception rised when an invalid file path is detected
    
    Parameters
    ----------
    path : str
        the invalid path
    """

    def __init__(self, path: str, *args: object) -> None:
        super().__init__(*args)
        self.__path = path

    def __str__(self) -> str:
        return """'{}' is an invalid file path.""".format(self.__path)


class UnknownFileExtension(Exception):
    """
    Exception rised when an unknown file extension is detected
    
    Parameters
    ----------
    extension : str
        the invalid extension
    """

    def __init__(self, extension: str, *args: object) -> None:
        super().__init__(*args)
        self.__extension = extension

    def __str__(self) -> str:
        return """The extension '{}' is not supported.""".format(self.__extension)


class InvalidSyntax(Exception):
    """
    Exception raised when an invalid circut string syntax is detected
    
    Parameters
    ----------
    msg : str
        the message containing the syntax error
    """
    def __init__(self, msg: str, *args: object) -> None:
        super().__init__(*args)
        self.__msg = remove_numbers(msg)

    def __str__(self) -> str:
        return """Syntax error in circuit string:\n{}""".format(self.__msg)

class InvalidComponent(Exception):
    """
    Exception raised when an invalid component is used in the equivalent circuit representation

    Parameters
    ----------
    name : str
        the invalid component
    """
    def __init__(self, name: str, *args: object) -> None:
        super().__init__(*args)
        self.__name = remove_numbers(name)

    def __str__(self) -> str:
        return """The component type '{}' is not supported.""".format(self.__name)
    
class InvalidParametrization(Exception):
    """
    Exception raised when an mismatch in component naming is encountered in the definition of
    an equivalent circuit.

    Parameters
    ----------
    msg : str
        a message explaining the error
    """
    def __init__(self, msg: str, *args: object) -> None:
        super().__init__(*args)
        self.__msg = remove_numbers(msg)

    def __str__(self) -> str:
        return """{}""".format(self.__msg)