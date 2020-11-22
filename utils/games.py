import autograd.numpy as np
import autograd.scipy as sp
from scipy import stats


class WetChicken2D:

    """
    Benchmark reinforcement learning problem where 
    a canoist is paddling in a river upstream of a waterfall
    and is rewarded for approaching it without being sent over.
    The canoeist can paddle upstream or side to side.
    The river has current (deterministic) and turbulence (stochastic),
    which vary along the width of the river (i.e. there is greater
    turbulence but slower flow along the river's left bank).

    This implementation is a simplified 2D version of the problem
    in discrete space. The rivers is represented as a grid,
    and current and flow are encoded in the transition
    probabilities between cells.

    The original Wet Chicken problem (in 1 dimension)
    is attributed to Volker Tresp in a 1994 Technical Report.
    Hans & Udluf (2009) state a 2D version (in continuous space):
    https://www.tu-ilmenau.de/fileadmin/media/neurob/publications/conferences_int/2009/Hans-ICANN-2009.pdf 
    """

    def __init__(self):

        pass

