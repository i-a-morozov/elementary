"""
Dipole
------

Dipole element factory

"""
from typing import Optional
from typing import Callable

import jax
from jax import Array

from elementary.element import element_factory


def curvature(s:Array, r:Array) -> Array:
    """
    Curvature

    """
    return 1/r


def vector(qs:Array, s:Array, r:Array) -> tuple[Array, Array, Array]:
    """
    Vector potential

    """
    q_x, *_ = qs
    a_x, a_y, a_s = jax.numpy.zeros_like(qs)
    a_s = (-1/2*q_x**2/(q_x + r) - (q_x*r)/(q_x + r))/r
    return a_x, a_y, a_s


def dipole_factory(beta:Optional[float]=None,
                   gamma:Optional[float]=None,
                   driver:Optional[Callable[..., Array]]=None,
                   settings:Optional[dict]=None,
                   order:int=0,
                   iterations:int=1) -> Callable[..., Array]:
    """
    Dipole element transfer map

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
                              curvature=curvature,
                              beta=beta,
                              gamma=gamma,
                              driver=driver,
                              settings=settings,
                              order=order,
                              iterations=iterations,
                              autonomous=True)
    def dipole(qsps, length, angle):
        return element(qsps, length, 0.0, length/angle)
    return dipole
