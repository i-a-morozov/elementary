"""
Alignment
---------

Alignment transformations factory

"""
from typing import Optional
from typing import Callable
import jax
from jax import Array

def alignment_factory(beta:Optional[float]=None,
                      gamma:Optional[float]=None,
                      flag:bool=False) -> tuple[Callable[..., Array], Callable[..., Array]]:
    """
    Generate entrance and exit alignment error transformations

    Parameters
    ----------
    beta: Optional[float]
        beta
    gamma: Optional[float]
        gamma
    flag: bool, default=False
        non-zero layout angle flag

    Returns
    -------
    tuple[Callable[..., Array], Callable[..., Array]]

    """
    beta = beta if beta else 1.0
    constant = 1/(beta**2*gamma**2) if gamma else 0.0

    def xyz_entrance(qsps:Array,
                    dx:Array,
                    dy:Array,
                    dz:Array,
                    wx:Array,
                    wy:Array,
                    wz:Array) -> Array:
        """
        Apply entrance translation & rotation alignment errors

        Parameters
        ----------
        qsps: Array
            initial state
        dx: Array
            dx
        dy: Array
            dy
        dz: Array
            dz
        wx: Array
            wx
        wy: Array
            wy
        wz: Array
            wz

        Returns
        -------
        Array

        """
        qsps = tx(qsps, +dx)
        qsps = ty(qsps, +dy)
        qsps = tz(qsps, +dz, beta, constant)
        qsps = rx(qsps, +wx, beta, constant)
        qsps = ry(qsps, +wy, beta, constant)
        qsps = rz(qsps, +wz)
        return qsps

    def xyz_exit(qsps:Array,
                dx:Array,
                dy:Array,
                dz:Array,
                wx:Array,
                wy:Array,
                wz:Array,
                length:Array,
                angle:Array=0.0) -> Array:
        """
        Apply exit translation & rotation alignment errors

        Parameters
        ----------
        qsps: Array
            initial state
        dx: Array
            dx
        dy: Array
            dy
        dz: Array
            dz
        wx: Array
            wx
        wy: Array
            wy
        wz: Array
            wz
        length: Array
            layout block length
        angle: Array
            layout block angle

        Returns
        -------
        Array

        """
        if flag:
            qsps = ry(qsps, +angle/2, beta, constant)
            qsps = tz(qsps, -2.0*length/angle*jax.numpy.sin(angle/2.0), beta, constant)
            qsps = ry(qsps, +angle/2, beta, constant)
        else:
            qsps = tz(qsps, -length, beta=beta, constant=constant)
        qsps = rz(qsps, -wz)
        qsps = ry(qsps, -wy, beta, constant)
        qsps = rx(qsps, -wx, beta, constant)
        qsps = tz(qsps, -dz, beta, constant)
        qsps = ty(qsps, -dy)
        qsps = tx(qsps, -dx)
        if flag:
            qsps = ry(qsps, -angle/2, beta, constant)
            qsps = tz(qsps, +2.0*length/angle*jax.numpy.sin(angle/2.0), beta, constant)
            qsps = ry(qsps, -angle/2, beta, constant)
        else:
            qsps = tz(qsps, +length, beta, constant)
        return qsps

    return xyz_entrance, xyz_exit


def tx(qsps:Array, dx:Array, *args:Array) -> Array:
    """
    TX translation (sign matches MADX)

    Parameters
    ----------
    qsps: Array
        initial state
    dx: Array
        q_x translation error

    Returns
    -------
    Array

    """
    q_x, q_y, q_s, p_x, p_y, p_s = qsps
    Q_x = q_x - dx
    Q_y = q_y
    Q_s = q_s
    P_x = p_x
    P_y = p_y
    P_s = p_s
    return jax.numpy.stack([Q_x, Q_y, Q_s, P_x, P_y, P_s])


def ty(qsps:Array, dy:Array, *args:Array) -> Array:
    """
    TY translation (sign matches MADX)

    Parameters
    ----------
    qsps: Array
        initial state
    dy: Array
        q_y translation error

    Returns
    -------
    Array

    """
    q_x, q_y, q_s, p_x, p_y, p_s = qsps
    Q_x = q_x
    Q_y = q_y - dy
    Q_s = q_s
    P_x = p_x
    P_y = p_y
    P_s = p_s
    return jax.numpy.stack([Q_x, Q_y, Q_s, P_x, P_y, P_s])


def tz(qsps:Array, dz:Array, beta:float=1.0, constant:float=0.0) -> Array:
    """
    TZ translation (sign matches MADX)

    Parameters
    ----------
    qsps: Array
        initial state
    dz: Array
        q_s translation error
    beta: float, default=1.0
        beta
    constant: float, default=0.0
        1/(beta*gamma)**2

    Returns
    -------
    Array

    """
    q_x, q_y, q_s, p_x, p_y, p_s = qsps
    sqrt = jax.numpy.sqrt((1/beta + p_s)**2 - p_x**2 - p_y**2 - constant)
    Q_x = q_x + p_x*dz/sqrt
    Q_y = q_y + p_y*dz/sqrt
    Q_s = q_s + dz/beta - dz*(1/beta + p_s)/sqrt
    P_x = p_x
    P_y = p_y
    P_s = p_s
    return jax.numpy.stack([Q_x, Q_y, Q_s, P_x, P_y, P_s])


def rx(qsps:Array, wx:Array, beta:float=1.0, constant:float=0.0) -> Array:
    """
    RX rotation (sign matches MADX)

    Parameters
    ----------
    qsps: Array
        initial state
    wx: Array
        q_x rotation angle
    beta: float, default=1.0
        beta
    constant: float, default=0.0
        1/(beta*gamma)**2

    Returns
    -------
    Array

    """
    q_x, q_y, q_s, p_x, p_y, p_s = qsps
    sqrt = jax.numpy.sqrt((1/beta + p_s)**2 - p_x**2 - p_y**2 - constant)
    cos = jax.numpy.cos(-wx)
    sin = jax.numpy.sin(-wx)
    tan = sin/cos
    Q_x = q_x + p_x*q_y*tan/(sqrt - p_y*tan)
    Q_y = q_y/cos/(1 - p_y*tan/sqrt)
    Q_s = q_s - (1/beta + p_s)*q_y*tan/(sqrt - p_y*tan)
    P_x = p_x
    P_y = p_y*cos + sqrt*sin
    P_s = p_s
    return jax.numpy.stack([Q_x, Q_y, Q_s, P_x, P_y, P_s])


def ry(qsps:Array, wy:Array, beta:float=1.0, constant:float=0.0) -> Array:
    """
    RY rotation (sign matches MADX)

    Parameters
    ----------
    qsps: Array
        initial state
    wy: Array
        q_y rotation angle
    beta: float, default=1.0
        beta
    constant: float, default=0.0
        1/(beta*gamma)**2

    Returns
    -------
    Array

    """
    q_x, q_y, q_s, p_x, p_y, p_s = qsps
    sqrt = jax.numpy.sqrt((1/beta + p_s)**2 - p_x**2 - p_y**2 - constant)
    cos = jax.numpy.cos(-wy)
    sin = jax.numpy.sin(-wy)
    tan = sin/cos
    Q_x = q_x/cos/(1 - p_x/sqrt*tan)
    Q_y = q_y + p_y*q_x*tan/(sqrt - p_x*tan)
    Q_s = q_s - (1/beta + p_s)*q_x*tan/(sqrt - p_x*tan)
    P_x = p_x*cos + sqrt*sin
    P_y = p_y
    P_s = p_s
    return jax.numpy.stack([Q_x, Q_y, Q_s, P_x, P_y, P_s])


def rz(qsps:Array, wz:Array) -> Array:
    """
    RZ rotation (sign matches MADX)

    Parameters
    ----------
    qsps: Array
        initial state
    wz: Array
        q_z rotation angle

    Returns
    -------
    Array

    """
    q_x, q_y, q_s, p_x, p_y, p_s = qsps
    cos = jax.numpy.cos(wz)
    sin = jax.numpy.sin(wz)
    Q_x = q_x*cos + q_y*sin
    Q_y = q_y*cos - q_x*sin
    Q_s = q_s
    P_x = p_x*cos + p_y*sin
    P_y = p_y*cos - p_x*sin
    P_s = p_s
    return jax.numpy.stack([Q_x, Q_y, Q_s, P_x, P_y, P_s])
