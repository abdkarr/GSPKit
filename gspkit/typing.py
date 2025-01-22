from enum import Enum
from typing import Optional

import numpy as np

RNGType = Optional[np.random.Generator | int]
""" 
The type alias for random number generator. 

- If an `#!python int`, a random number generator whose seed number is set to
  the given number is created. 
- If `#!python np.random.Generator`, it is used as the random number generator.
- If `#!python None`, a random generator without setting any seed number is
  created.
"""

class FilterType(Enum):
    r"""
    Enum class for available graph filters, which are listed below:
    """

    GAUSSIAN = "gaussian"
    """ 
    Gaussian filter for graphs implemented with graph Laplacian. See
    [gspkit.filters.gaussian][] the for filter defition.
    """

    TIKHONOV = "tikhonov" 
    """
    Tikhonov filter for graphs implemented with graph Laplacian. See
    [gspkit.filters.tikhonov][] the for filter defition.
    """

    HEAT = "heat"
    """
    Heat kernel filter for graphs implemented with graph Laplacian. See
    [gspkit.filters.heat][] the for filter defition.
    """

class GraphModelType(Enum):
    r"""
    Enum class for available graph models, which are listed below:
    """

    ER = "er"
    r"""
    Erdős-Rényi graph. The model details can be found
    [gspkit.data.graphs.gen_er][].  
    """

    BA = "ba"
    r"""
    Barabási–Albert graph. The model details can be found
    [gspkit.data.graphs.gen_ba][].  
    """

    RGG = "rgg"
    r"""
    Random Geometric Graph. The model details can be found
    [gspkit.data.graphs.gen_rgg][].  
    """

    SIGNED_ER = "signed-er"
    r"""
    Signed Erdős-Rényi graph. Implementation details can be at
    [gspkit.data.graphs.gen_signed_er]
    """

    SIGNED_BA = "signed-ba"
    r"""
    Signed Barabási–Albert graph. Implementation details can be at
    [gspkit.data.graphs.gen_signed_ba]
    """