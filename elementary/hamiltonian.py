"""
Hamiltonian
-----------

Generic single particle Hamiltonian factory

"""
from typing import Callable
from typing import Optional

import jax
from jax import Array


def hamiltonian_factory(vector:Callable[..., tuple[Array, Array, Array]],
                        scalar:Optional[Callable[..., Array]]=None, *,
                        curvature:Optional[Callable[..., Array]]=None,
                        torsion:Optional[Callable[..., Array]]=None,
                        beta:Optional[float]=None,
                        gamma:Optional[float]=None) -> Callable[..., Array]:
    """
    Generic single particle Hamiltonian factory

    H = H(qs, ps; s) = H(q_x, q_y, q_s, p_x, p_y, p_s; s)
    H = p_s/beta - torsion(s)*(q_x*p_y - q_y*p_x) - (1 + curvature(s)*q_x)*(root + a_s(qs, s))
    root = sqrt(P_s**2 - P_x**2 - P_y**2 - 1/(beta*gamma)**2)
    P_s = p_s + 1/beta - scalar(qs, s)
    P_x = p_x - a_x(qs, s)
    P_y = p_y - a_y(qs, s)

    Parameters
    ----------
    vector: Callable[..., tuple[Array, Array, Array]]
        normalized vector potential
    scalar: Optional[Callable[..., Array]]
        normalized scalar potential
    curvature: Optional[Callable[..., Array]]
        curvature
    torsion: Optional[Callable[..., Array]]
        torsion
    beta: Optional[float]
        beta
    gamma: Optional[float]
        gamma

    Returns
    -------
    Callable[..., Array]

    Note
    ----
    Vector and scalar potentials are assumed to have (qs, *args) signatures
    Curvature and torsion are functions of independent parameter
    The resulting hamiltonian has (qs, ps, s, *args) signature

    """
    beta = beta if beta else 1.0
    constant = 1/(beta**2*gamma**2) if gamma else 0.0
    def hamiltonian(qs: Array, ps: Array, s: Array, *args: Array) -> Array:
        q_x, q_y, *_ = qs
        p_x, p_y, p_s = ps
        a_x, a_y, a_s = vector(qs, s, *args)
        P_s = p_s + 1/beta
        if scalar:
            P_s = P_s - scalar(qs, s, *args)
        P_x = p_x - a_x
        P_y = p_y - a_y
        root = jax.numpy.sqrt(P_s**2 - P_x**2 - P_y**2 - constant)
        factor = 1.0
        if curvature:
            factor = 1 + curvature(s, *args)*q_x
        result = p_s/beta - factor*(root + a_s)
        if torsion:
            result = result - torsion(s, *args)*(q_x*p_y - q_y*p_x)
        return result
    return hamiltonian


def autonomize(hamiltonian:Callable[..., Array]) -> Callable[..., Array]:
    """
    Autonomize hamiltonian

    Parameters
    ----------
    hamiltonian: Callable[..., Array]
        input hamiltonian

    Returns
    -------
    Callable[..., Array]

    """
    def autonomous(qs: Array, ps: Array, s: Array, *args: Array) -> Array:
        q_x, q_y, q_s, q_t = qs
        p_x, p_y, p_s, p_t = ps
        Qs = jax.numpy.stack([q_x, q_y, q_s])
        Ps = jax.numpy.stack([p_x, p_y, p_s])
        return hamiltonian(Qs, Ps, q_t, *args) + p_t
    return autonomous
