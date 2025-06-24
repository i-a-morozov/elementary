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

def dipole_factory(multipole:bool=False,
                   beta:Optional[float]=None,
                   gamma:Optional[float]=None,
                   driver:Optional[Callable[..., Array]]=None,
                   settings:Optional[dict]=None,
                   order:int=0,
                   iterations:int=1,
                   final:bool=True) -> Callable[..., Array]:
    """
    Dipole element transfer map

    Parameters
    ----------
    multipole: bool, default=False
        multipole flag
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
    def vector(qs:Array, s:Array, r:Array) -> tuple[Array, Array, Array]:
        return vector_dipole(qs, s, r)
    if multipole:
        def vector(qs:Array,
                   s:Array,
                   r:Array,
                   kq_n:Array,
                   kq_s:Array,
                   ks_n:Array,
                   ks_s:Array,
                   ko_n:Array,
                   ko_s:Array) -> tuple[Array, Array, Array]:
            a_x, a_y, a_s = vector_dipole(qs, s, r)
            A_x, A_y, A_s = vector_quadrupole(qs, s, r, kq_n, kq_s)
            a_x, a_y, a_s = a_x + A_x, a_y + A_y, a_s + A_s
            A_x, A_y, A_s = vector_sextupole(qs, s, r, ks_n, ks_s)
            a_x, a_y, a_s = a_x + A_x, a_y + A_y, a_s + A_s
            A_x, A_y, A_s = vector_octupole(qs, s, r, ko_n, ko_s)
            a_x, a_y, a_s = a_x + A_x, a_y + A_y, a_s + A_s
            return a_x, a_y, a_s
    element = element_factory(vector,
                              scalar=None,
                              curvature=curvature,
                              beta=beta,
                              gamma=gamma,
                              driver=driver,
                              settings=settings,
                              order=order,
                              iterations=iterations,
                              autonomous=True,
                              final=final)
    def dipole(qsps, length, angle, *args):
        r = jax.numpy.abs(length)/angle
        return element(qsps, length, 0.0, r, *args)
    return dipole


def curvature(s:Array, r:Array, *args) -> Array:
    """
    Curvature

    """
    return 1/r


def vector_dipole(qs:Array, s:Array, r:Array) -> tuple[Array, Array, Array]:
    """
    Vector potential

    """
    q_x, *_ = qs
    a_x, a_y, a_s = jax.numpy.zeros_like(qs)
    a_s = (-1/2*q_x**2/(q_x + r) - q_x*r/(q_x + r))/r
    return a_x, a_y, a_s


def vector_quadrupole(qs:Array, s:Array, r:Array, kn:Array, ks:Array) -> tuple[Array, Array, Array]:
    """
    Cylindrical quadrupole potential

    """
    q_x, q_y, _ = qs
    a_x, a_y, a_s = jax.numpy.zeros_like(qs)
    a_s = a_s + kn/(1 + q_x/r)*(
        -q_x**2/2.0 + q_y**2/2.0 +
        q_x**6*q_y**4/(24.*r**8) -
        q_x**4*q_y**6/(16.*r**8) +
        15*q_x**2*q_y**8/(896.*r**8) -
        q_y**10/(2304.*r**8) -
        q_x**5*q_y**4/(24.*r**7) +
        q_x**3*q_y**6/(24.*r**7) -
        5*q_x*q_y**8/(896.*r**7) +
        q_x**4*q_y**4/(24.*r**6) -
        q_x**2*q_y**6/(40.*r**6) +
        q_y**8/(896.*r**6) -
        q_x**3*q_y**4/(24.*r**5) +
        q_x*q_y**6/(80.*r**5) +
        q_x**2*q_y**4/(24.*r**4) -
        q_y**6/(240.*r**4) -
        q_x*q_y**4/(24.*r**3) +
        q_y**4/(24.*r**2) -
        q_x**3/(3.*r) +
        q_x*q_y**2/(2.*r)
    )
    a_s = a_s + ks/(1 + q_x/r)*(
        q_x*q_y +
        q_x**7*q_y**3/(6.*r**8) -
        21*q_x**5*q_y**5/(40.*r**8) +
        5*q_x**3*q_y**7/(16.*r**8) -
        35*q_x*q_y**9/(1152.*r**8) -
        q_x**6*q_y**3/(6.*r**7) +
        3*q_x**4*q_y**5/(8.*r**7) -
        15*q_x**2*q_y**7/(112.*r**7) +
        5*q_y**9/(1152.*r**7) +
        q_x**5*q_y**3/(6.*r**6) -
        q_x**3*q_y**5/(4.*r**6) +
        5*q_x*q_y**7/(112.*r**6) -
        q_x**4*q_y**3/(6.*r**5) +
        3*q_x**2*q_y**5/(20.*r**5) -
        q_y**7/(112.*r**5) +
        q_x**3*q_y**3/(6.*r**4) -
        3*q_x*q_y**5/(40.*r**4) -
        q_x**2*q_y**3/(6.*r**3) +
        q_y**5/(40.*r**3) +
        q_x*q_y**3/(6.*r**2) +
        q_x**2*q_y/r -
        q_y**3/(6.*r)
    )
    return a_x, a_y, a_s


def vector_sextupole(qs:Array, s:Array, r:Array, kn:Array, ks:Array) -> tuple[Array, Array, Array]:
    """
    Cylindrical sextupole potential

    """
    q_x, q_y, _ = qs
    a_x, a_y, a_s = jax.numpy.zeros_like(qs)
    a_s = a_s + kn/2.0/(1 + q_x/r)*(
        -q_x**3/3. +
        q_x*q_y**2 -
        (q_x**6*q_y**4)/(12.*r**7) +
        (q_x**4*q_y**6)/(8.*r**7) -
        (15*q_x**2*q_y**8)/(448.*r**7) +
        q_y**10/(1152.*r**7) +
        (q_x**5*q_y**4)/(12.*r**6) -
        (q_x**3*q_y**6)/(12.*r**6) +
        (5*q_x*q_y**8)/(448.*r**6) -
        (q_x**4*q_y**4)/(12.*r**5) +
        (q_x**2*q_y**6)/(20.*r**5) -
        q_y**8/(448.*r**5) +
        (q_x**3*q_y**4)/(12.*r**4) -
        (q_x*q_y**6)/(40.*r**4) -
        (q_x**2*q_y**4)/(12.*r**3) +
        q_y**6/(120.*r**3) +
        (q_x*q_y**4)/(12.*r**2) -
        q_x**4/(4.*r) +
        (q_x**2*q_y**2)/r -
        q_y**4/(12.*r)
    )
    a_s = a_s + ks/2.0/(1 + q_x/r)*(
        q_x**2*q_y -
        q_y**3/3. -
        (q_x**7*q_y**3)/(6.*r**7) +
        (11*q_x**5*q_y**5)/(20.*r**7) -
        (37*q_x**3*q_y**7)/(112.*r**7) +
        (65*q_x*q_y**9)/(2016.*r**7) +
        (q_x**6*q_y**3)/(6.*r**6) -
        (2*q_x**4*q_y**5)/(5.*r**6) +
        (81*q_x**2*q_y**7)/(560.*r**6) -
        (19*q_y**9)/(4032.*r**6) -
        (q_x**5*q_y**3)/(6.*r**5) +
        (11*q_x**3*q_y**5)/(40.*r**5) -
        (q_x*q_y**7)/(20.*r**5) +
        (q_x**4*q_y**3)/(6.*r**4) -
        (7*q_x**2*q_y**5)/(40.*r**4) +
        (3*q_y**7)/(280.*r**4) -
        (q_x**3*q_y**3)/(6.*r**3) +
        (q_x*q_y**5)/(10.*r**3) +
        (q_x**2*q_y**3)/(6.*r**2) -
        q_y**5/(20.*r**2) +
        (q_x**3*q_y)/r -
        (2*q_x*q_y**3)/(3.*r)
    )
    return a_x, a_y, a_s


def vector_octupole(qs:Array, s:Array, r:Array, kn:Array, ks:Array) -> tuple[Array, Array, Array]:
    """
    Cylindrical octupole potential

    """
    q_x, q_y, _ = qs
    a_x, a_y, a_s = jax.numpy.zeros_like(qs)
    a_s = a_s + kn/6.0/(1 + q_x/r)*(
        -q_x**4/4.0 +
        (3*q_x**2*q_y**2)/2. -
        q_y**4/4. +
        (q_x**6*q_y**4)/(8.*r**6) -
        (q_x**4*q_y**6)/(5.*r**6) +
        (243*q_x**2*q_y**8)/(4480.*r**6) -
        (19*q_y**10)/(13440.*r**6) -
        (q_x**5*q_y**4)/(8.*r**5) +
        (11*q_x**3*q_y**6)/(80.*r**5) -
        (3*q_x*q_y**8)/(160.*r**5) +
        (q_x**4*q_y**4)/(8.*r**4) -
        (7*q_x**2*q_y**6)/(80.*r**4) +
        (9*q_y**8)/(2240.*r**4) -
        (q_x**3*q_y**4)/(8.*r**3) +
        (q_x*q_y**6)/(20.*r**3) +
        (q_x**2*q_y**4)/(8.*r**2) -
        q_y**6/(40.*r**2) -
        q_x**5/(5.*r) +
        (3*q_x**3*q_y**2)/(2.*r) -
        (q_x*q_y**4)/(2.*r)
    )
    a_s = a_s + ks/6.0/(1 + q_x/r)*(
        q_x**3*q_y -
        q_x*q_y**3 +
        (q_x**7*q_y**3)/(6.*r**6) -
        (3*q_x**5*q_y**5)/(5.*r**6) +
        (41*q_x**3*q_y**7)/(112.*r**6) -
        (145*q_x*q_y**9)/(4032.*r**6) -
        (q_x**6*q_y**3)/(6.*r**5) +
        (9*q_x**4*q_y**5)/(20.*r**5) -
        (93*q_x**2*q_y**7)/(560.*r**5) +
        (11*q_y**9)/(2016.*r**5) +
        (q_x**5*q_y**3)/(6.*r**4) -
        (13*q_x**3*q_y**5)/(40.*r**4) +
        (17*q_x*q_y**7)/(280.*r**4) -
        (q_x**4*q_y**3)/(6.*r**3) +
        (9*q_x**2*q_y**5)/(40.*r**3) -
        q_y**7/(70.*r**3) +
        (q_x**3*q_y**3)/(6.*r**2) -
        (3*q_x*q_y**5)/(20.*r**2) +
        (q_x**4*q_y)/r -
        (3*q_x**2*q_y**3)/(2.*r) +
        q_y**5/(10.*r)
    )
    return a_x, a_y, a_s
