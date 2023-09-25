from typing import Any, Dict, List

def elements_identical(li: List) -> bool:
    """Return true iff all elements of li are identical."""
    if len(li) == 0:
        return True

    return li.count(li[0]) == len(li)


def filter_none(li: List) -> List:
    """Remove all None elements from li."""
    return [i for i in li if (i is not None)]


def list_dict_swap(v: List[Dict[Any, Any]]) -> Dict[Any, List[Any]]:
    """Convert list of dicts to a dict of lists.
    >>> list_dict_swap([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]) == {'a': [1, 3], 'b': [2, 4]}
    """
    return {k: [dic[k] for dic in v] for k in v[0]}


def transpose_lists(li: List[List[Any]]) -> List[List[Any]]:
    """Transpose list of lists as if it were a matrix."""
    return list(map(list, zip(*li)))


def flatten_top(li: List[List[Any]]) -> List[Any]:
    """Flatten one level of a nested list."""
    return [item for sl in li for item in sl]
