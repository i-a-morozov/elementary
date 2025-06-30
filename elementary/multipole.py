"""
Multipole
---------

Multipole element factory

"""
from typing import Optional
from typing import Callable

import jax
from jax import Array

from elementary.element import element_factory


def multipole_factory(beta:Optional[float]=None,
                      gamma:Optional[float]=None,
                      driver:Optional[Callable[..., Array]]=None,
                      settings:Optional[dict]=None,
                      order:int=0,
                      iterations:int=1,
                      final:bool=True) -> Callable[..., Array]:
    """
    Multipole element transfer map

    Parameters
    ----------
    beta: Optional[float]
        relativistic beta
    gamma: Optional[float]
        relativistic gamma
    driver: Optional[Callable[..., Array]]
        symplectic integrator (tao)
    settings: Optional[dict]
        configuration settings for integrator
    order: int, default=0
        yoshida composition order
    iterations: int, default=1
        number of integration
    final: bool, default=True
        flag to return only the final state

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
                              autonomous=True,
                              final=final)
    def multipole(qsps, length, kq_n, kq_s, ks_n, ks_s, ko_n, ko_s):
        return element(qsps, length, 0.0, kq_n, kq_s, ks_n, ks_s, ko_n, ko_s)
    return multipole

def vector(qs:Array,
           s:Array,
           kq_n:Array,
           kq_s:Array,
           ks_n:Array,
           ks_s:Array,
           ko_n:Array,
           ko_s:Array) -> tuple[Array, Array, Array]:
    """
    Vector potential

    """
    q_x, q_y, _ = qs
    a_x, a_y, a_s = jax.numpy.zeros_like(qs)
    a_s = a_s - 1/2*kq_n*(q_x**2 - q_y**2) + kq_s*q_x*q_y
    a_s = a_s - ks_n/2*(q_x**3/3 - q_x*q_y**2) - ks_s/2*(-q_x**2*q_y + q_y**3/3)
    a_s = a_s - ko_n/6*(q_x**4/4 - 3*q_x**2*q_y**2/2 + q_y**4/4) - ko_s/6*(-q_x**3*q_y + q_x*q_y**3)
    return a_x, a_y, a_s
