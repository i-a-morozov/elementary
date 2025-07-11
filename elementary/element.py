"""
Element
-------

Generic element factory

"""
from typing import Callable
from typing import Optional

import jax
from jax import Array

from elementary import fold
from elementary import nest
from elementary import nest_list
from elementary import tao
from elementary import sequence

from elementary.hamiltonian import hamiltonian_factory
from elementary.hamiltonian import autonomize


def element_factory(vector:Optional[Callable[..., tuple[Array, Array, Array]]]=None,
                    scalar:Optional[Callable[..., Array]]=None,
                    curvature:Optional[Callable[..., Array]]=None,
                    torsion:Optional[Callable[..., Array]]=None,
                    hamiltonian:Optional[Callable[..., Array]]=None,
                    beta:Optional[float]=None,
                    gamma:Optional[float]=None,
                    driver:Optional[Callable[..., Array]]=None,
                    settings:Optional[dict]=None,
                    order:int=0,
                    iterations:int=1,
                    autonomous:bool=True,
                    final:bool=True) -> Callable[..., Array]:
    """
    Generate generic element transfer map

    Parameters
    ----------
    vector: Optional[Callable[..., tuple[Array, Array, Array]]]
        normalized vector potential
    scalar: Optional[Callable[..., Array]]
        normalized scalar potential
    curvature: Optional[Callable[..., Array]]
        curvature
    torsion: Optional[Callable[..., Array]]
        torsion
    hamiltonian: Optional[Callable[..., Array]]
        generic hamiltonian (other callables are ignored)
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
    autonomous: bool, default=True
        autonomous flag
    final: bool, default=True
        flag to return only the final state

    Returns
    -------
    Callable[..., Array]

    """
    if hamiltonian is None:
        hamiltonian = hamiltonian_factory(
            vector=vector,
            scalar=scalar,
            curvature=curvature,
            torsion=torsion,
            beta=beta,
            gamma=gamma
        )
    if vector is None:
        def vector(qs:Array, s:Array, *args:Array) -> tuple[Array, Array, Array]:
            return tuple(jax.numpy.zeros_like(qs))
    if autonomous:
        table = [(driver if driver else tao)(hamiltonian, **settings if settings else {})]
        slice = fold(sequence(0, order, table, merge=False))
        if final:
            def element(qsps:Array, length:Array, start:Array, *args:Array) -> Array:
                return nest(iterations, slice)(qsps, length/iterations, start, *args)
            return element
        def element(qsps:Array, length:Array, start:Array, *args:Array) -> Array:
            return nest_list(iterations, slice)(qsps, length/iterations, start, *args)
        return element
    extended = autonomize(hamiltonian)
    table = [(driver if driver else tao)(extended, **settings if settings else {})]
    slice = fold(sequence(0, order, table, merge=False))
    if final:
        def element(qsps:Array, length:Array, start:Array, *args:Array) -> Array:
            qs, ps = jax.numpy.reshape(qsps, (2, -1))
            q_t = start
            p_t = -hamiltonian(qs, ps, start, *args)
            qs = jax.numpy.concat([qs, q_t.reshape(-1)])
            ps = jax.numpy.concat([ps, p_t.reshape(-1)])
            qsps = jax.numpy.hstack([qs, ps])
            qsps = nest(iterations, slice)(qsps, length/iterations, start, *args)
            q_x, q_y, q_s, _, p_x, p_y, p_s, _ = qsps
            return jax.numpy.stack([q_x, q_y, q_s, p_x, p_y, p_s])
        return element
    def element(qsps:Array, length:Array, start:Array, *args:Array) -> Array:
        qs, ps = jax.numpy.reshape(qsps, (2, -1))
        q_t = start
        p_t = -hamiltonian(qs, ps, start, *args)
        qs = jax.numpy.concat([qs, q_t.reshape(-1)])
        ps = jax.numpy.concat([ps, p_t.reshape(-1)])
        qsps = jax.numpy.hstack([qs, ps])
        qsps = nest_list(iterations, slice)(qsps, length/iterations, start, *args)
        q_x, q_y, q_s, _, p_x, p_y, p_s, _ = qsps.T
        return jax.numpy.stack([q_x, q_y, q_s, p_x, p_y, p_s]).T
    return element
