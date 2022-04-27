from copy import deepcopy
from typing import Any, Dict, List, Union

# Nested config objects are delimited with this character
DEFAULT_NEST_DELIMITER = "."


def nest_config(params: Dict, delimiter: str = DEFAULT_NEST_DELIMITER) -> Dict:
    # Deepcopy to prevent modifying the original params object
    params_copy: Dict = deepcopy(params)
    _unflatten_dict(params_copy, delimiter)
    return params_copy


def unnest_config(params: Dict, delimiter: str = DEFAULT_NEST_DELIMITER) -> Dict:
    # Deepcopy to prevent modifying the original params object
    params_copy: Dict = deepcopy(params)
    _flatten_dict(params_copy, delimiter)
    return params_copy


def _flatten_dict(d: Dict, delimiter: str) -> None:
    """Flatten dict with nested keys into a single level dict with a specified delimiter.
    Based on community solution:
    https://github.com/wandb/client/issues/982#issuecomment-1014525666
    """
    if type(d) == dict:
        for k, v in list(d.items()):
            if type(v) == dict:
                _flatten_dict(v, delimiter)
                d.pop(k)
                if not isinstance(k, str):
                    raise ValueError(
                        f"Config keys must be strings, found {k} of type {type(k)}"
                    )
                for subkey, subval in v.items():
                    if not isinstance(subkey, str):
                        raise ValueError(
                            f"Config keys must be strings, found {subkey} of type {type(subkey)}"
                        )
                    d[f"{k}{delimiter}{subkey}"] = subval


def _unflatten_dict(d: Dict, delimiter: str) -> None:
    """Un-flatten a single level dict to a nested dict with a specified delimiter.
    Based on community solution:
    https://github.com/wandb/client/issues/982#issuecomment-1014525666
    """
    if type(d) == dict:
        # The reverse sorting here ensures that "foo.bar" will appear before "foo"
        for k in sorted(d.keys(), reverse=True):
            if not isinstance(k, str):
                raise ValueError(
                    f"Config keys must be strings, found {k} of type {type(k)}"
                )
            if delimiter in k:
                subdict: Union[Any, Dict] = d
                subkeys: List[str] = k.split(delimiter)
                for i, subkey in enumerate(subkeys[:-1]):
                    if subkey in subdict:
                        subdict = subdict[subkey]
                        if not isinstance(subdict, dict):
                            conflict_key: str = delimiter.join(subkeys[: i + 1])
                            raise ValueError(
                                f"While nesting config, found key {subkey} which conflics with key {conflict_key}"
                            )
                    else:
                        # Create a nested dictionary under the parent key
                        _d: Dict = dict()
                        subdict[subkey] = _d
                        subdict = _d
                if isinstance(subdict, dict):
                    subdict[subkeys[-1]] = d.pop(k)
