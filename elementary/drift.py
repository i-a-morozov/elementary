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


def drift_factory(exact:bool=True,
                  beta:Optional[float]=None,
                  gamma:Optional[float]=None,
                  driver:Optional[Callable[..., Array]]=None,
                  settings:Optional[dict]=None,
                  order:int=0,
                  iterations:int=1,
                  final:bool=True) -> Callable[..., Array]:
    """
    Drift element transfer map

    Parameters
    ----------
    exact: bool, default=False
        ideal transfromation
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
    final: bool, default=True
        flag to return only the final state

    Returns
    -------
    Callable[..., Array]

    """
    if exact:
        beta = beta if beta else 1.0
        constant = 1/(beta**2*gamma**2) if gamma else 0.0
        def drift(qsps:Array, length:Array) -> Array:
            return mapping(qsps, length, beta, constant)
        return drift
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
    def drift(qsps, length):
        return element(qsps, length, 0.0)
    return drift

def vector(qs:Array, s:Array) -> tuple[Array, Array, Array]:
    """
    Vector potential

    """
    return tuple(jax.numpy.zeros_like(qs))


def mapping(qsps:Array, length:Array, beta:float=1.0, constant:float=0.0) -> Array:
    """
    Exact drift transformation

    """
    q_x, q_y, q_s, p_x, p_y, p_s = qsps
    dp = jax.numpy.sqrt((1/beta + p_s)**2 - p_x**2 - p_y**2 - constant)
    Q_x = q_x + p_x*length/dp
    Q_y = q_y + p_y*length/dp
    Q_s = q_s + length/beta - length*(1/beta + p_s)/dp
    P_x = p_x
    P_y = p_y
    P_s = p_s
    return jax.numpy.stack([Q_x, Q_y, Q_s, P_x, P_y, P_s])
