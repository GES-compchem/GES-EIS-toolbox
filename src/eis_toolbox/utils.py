def remove_numbers(string: str) -> str:
    """
    Remove all the numerical characters from a string

    Parameters
    ----------
    string: str
        the input string
    
    Returns
    -------
    str
        the string without numerical characters
    """
    x = string
    for i in range(10):
        x = x.replace(str(i), "")
    return x
