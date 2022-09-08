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