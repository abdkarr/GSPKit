import numpy as np

from gspkit import typing

def check_rng(rng: typing.RNG_TYPE) -> np.random.Generator:
    r"""Check a rng arguments for functions.

    The function can be used to check if `rng` argument is a valid `numpy`
    random number generator (RNG). If it is valid, it returns it as it is. If
    `rng` is `None`, it will create a RNG. If it is `int`, it will create a RNG
    with seed number set to the given int.

    Parameters
    ----------
    rng : None | int | np.random.Generator
        RNG to be checked.

    Returns
    -------
    np.random.Generator
        Returned RNG. 
    """
    
    # TODO Testing
    
    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, int):
        rng = np.random.default_rng(rng)
    
    return rng