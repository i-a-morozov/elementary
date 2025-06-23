"""
Drift
-----

Drift element factory

"""
from typing import Optional
from typing import Callable

import jax
from jax import Array

from elementary.element import element_factory


def drift_factory(beta:Optional[float]=None,
                  gamma:Optional[float]=None,
                  driver:Optional[Callable[..., Array]]=None,
                  settings:Optional[dict]=None,
                  order:int=0,
                  iterations:int=1) -> Callable[..., Array]:
    """
    Drift element transfer map

    Parameters
    ----------
    beta: Optional[float]
        relativistic beta
    gamma: Optional[float]
        relativistic gamma
    driver: Optional[Callable[..., Array]]
        symplectic integrator (midpoint)
    settings: Optional[dict]
        configuration settings for integrator
    order: int, default=0
        yoshida composition order
    iterations: int, default=1
        number of integration
    autonomous: bool, default=True
        autonomous flag

    Returns
    -------
    Callable[..., Array]

    """
    element = element_factory(vector,
                              scalar=None,
                              beta=beta,
                              gamma=gamma,
                              driver=driver,
                              settings=settings,
                              order=order,
                              iterations=iterations,
                              autonomous=True)
    def drift(qsps, length):
        return element(qsps, length, 0.0)
    return drift

def vector(qs:Array, s:Array) -> tuple[Array, Array, Array]:
    """
    Vector potential

    """
    return tuple(jax.numpy.zeros_like(qs))
