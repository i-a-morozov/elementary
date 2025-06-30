"""
Cavity
------

Cavity element factory

"""
from typing import Optional
from typing import Callable
from typing import Literal

import jax
from jax import Array

from elementary.util import CL
from elementary.util import bessel
from elementary.element import element_factory


def cavity_factory(rigidity:float,
                   kind:Literal['kick', 'main'] = 'kick',
                   beta:Optional[float]=None,
                   gamma:Optional[float]=None,
                   driver:Optional[Callable[..., Array]]=None,
                   settings:Optional[dict]=None,
                   order:int=0,
                   iterations:int=1,
                   final:bool=True,
                   epsilon:float=1.0E-15) -> Callable[..., Array]:
    """
    Cavity element transfer map

    Parameters
    ----------
    rigidity: float
        magnetic rigidity
    kind: Literal['kick', 'main'], default='kick'
        cavity type
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
    epsilon: float, default=1.0E-15
        epsilon

    Returns
    -------
    Callable[..., Array]

    """
    if kind == 'kick':
        def cavity(qsps, voltage, lag):
            q_x, q_y, q_s, p_x, p_y, p_s = qsps
            Q_x = q_x
            Q_y = q_y
            Q_s = q_s
            P_x = p_x
            P_y = p_y
            P_s = p_s + (1E6*voltage)/(rigidity*CL)*jax.numpy.sin(lag)
            return jax.numpy.stack([Q_x, Q_y, Q_s, P_x, P_y, P_s])
    if kind == 'main':
        def vector(qs:Array,
                   s:Array,
                   length:Array,
                   voltage:Array,
                   frequency:Array,
                   lag:Array) -> tuple[Array, Array, Array]:
            """
            Vector potential

            """
            q_x, q_y, q_s = qs
            a_x, a_y, a_s = jax.numpy.zeros_like(qs)
            omega = 2*jax.numpy.pi*(1E6*frequency)
            k = omega/CL
            factor = k*length/(2*beta)
            time = jax.numpy.sin(factor)/factor
            amplitude = (1E+6*voltage)/(length*time)
            r = jax.numpy.sqrt((q_x + epsilon)**2 + (q_y + epsilon)**2)
            a_s = amplitude/(rigidity*omega)*bessel(k*r)*jax.numpy.cos(k*(s/beta - q_s) + lag)
            return a_x, a_y, a_s
        element = element_factory(vector,
                                  scalar=None,
                                  beta=beta,
                                  gamma=gamma,
                                  driver=driver,
                                  settings=settings,
                                  order=order,
                                  iterations=iterations,
                                  autonomous=False,
                                  final=final)
        def cavity(qsps, length, voltage, frequency, lag):
            start = -length/2.0
            return element(qsps, length, start, length, voltage, frequency, lag)
    return cavity
