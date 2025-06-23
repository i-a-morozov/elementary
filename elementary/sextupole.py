"""
Sextupole
---------

Sextupole element factory

"""
from typing import Optional
from typing import Callable

import jax
from jax import Array

from elementary.element import element_factory


def sextupole_factory(beta:Optional[float]=None,
                      gamma:Optional[float]=None,
                      driver:Optional[Callable[..., Array]]=None,
                      settings:Optional[dict]=None,
                      order:int=0,
                      iterations:int=1) -> Callable[..., Array]:
    """
    Sextupole element transfer map

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
    def sextupole(qsps, length, kn, ks):
        return element(qsps, length, 0.0, kn, ks)
    return sextupole

def vector(qs:Array, s:Array, kn:Array, ks:Array) -> tuple[Array, Array, Array]:
    """
    Vector potential

    """
    q_x, q_y, _ = qs
    a_x, a_y, a_s = jax.numpy.zeros_like(qs)
    a_s = -kn/2*(q_x**3/3 - q_x*q_y**2) - ks/2*(-q_x**2*q_y + q_y**3/3)
    return a_x, a_y, a_s
