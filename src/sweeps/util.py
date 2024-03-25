import typing


def get_nested_value(obj: dict, key: typing.List[str]) -> typing.Any:
    """Get a nested value from a dictionary.

    Args:
        obj (dict): The dictionary to search.
        key (str): The key to search for.

    Returns:
        The value corresponding to the key. Raises a KeyError if the key is not found.
    """
    for subkey in key:
        obj = obj[subkey]
    return obj


def dict_has_nested_key(obj: dict, key: typing.List[str]) -> bool:
    """Check if a dictionary has a nested key.

    Args:
        obj (dict): The dictionary to search.
        key (str): The key to search for.

    Returns:
        True if the key is found, False otherwise.
    """
    try:
        get_nested_value(obj, key)
    except KeyError:
        return False
    else:
        return True
