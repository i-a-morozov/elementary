"""
Drift
-----

Drift element factory

"""
from typing import Optional
from typing import Callable

from functools import partial

import jax
from jax import Array

from elementary import fold
from elementary import nest
from elementary import midpoint
from elementary import sequence

from elementary.hamiltonian import hamiltonian_factory

def drift(length:Optional[Array]=None,
          beta:Optional[float]=None,
          gamma:Optional[float]=None,
          integrator:Optional[Callable[..., Array]]=None,
          settings:Optional[dict]=None,
          order:int=0,
          iterations:int=1) -> Callable[..., Array]:
    """
    Generate drift element transfer map

    Parameters
    ----------
    length: Optional[Array]
        length (parametric)
    beta: Optional[float]
        relativistic beta
    gamma: Optional[float]
        relativistic gamma
    integrator: Optional[Callable[..., Array]]
        symplectic integrator (midpoint)
    settings: Optional[dict]
        configuration settings for integrator
    order: int, default=0
        yoshida composition order
    iterations: int, default=1
        number of integration

    Returns
    -------
    Callable[..., Array]

    Note
    ----
    If length is not provided on construction, it is expected to be passed on call
    In this case, element transfer map is differentiable with respect to such parameter(s)

    If element parameters are specified on initializatin
    The returned element has 'rc' attribute containing construction parameters

    """
    hamiltonian = hamiltonian_factory(vector, scalar, beta=beta, gamma=gamma)
    data = [(integrator if integrator else midpoint)(hamiltonian, **settings if settings else {})]
    step = fold(sequence(0, order, data, merge=False))
    def closure(qsps, length=None):
        return nest(iterations, step)(qsps, length/iterations, 0.0)
    kwargs = {}
    if length:
        kwargs['length'] = length
    element = partial(closure, **kwargs)
    element.rc = kwargs
    return element


def vector(qs:Array, s:Array, *args:Array) -> Array:
    """
    Vector potential

    """
    return tuple(jax.numpy.zeros_like(qs))


def scalar(qs:Array, s:Array, *args:Array) -> Array:
    """
    Scalar potential

    """
    return jax.numpy.zeros_like(s)
