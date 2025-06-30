"""
Util
----

Utils module

"""
from os import system
from pathlib import Path

from math import factorial

import jax
from jax import Array

CL:float = 299792458.0
ME:float = 0.51099895069
MP:float = 938.27208943

def beta(gamma:float) -> float:
    """
    Compute relativistic beta

    """
    return (1 - 1/gamma**2)**0.5


def gamma(beta:float) -> float:
    """
    Compute relativistic gamma

    """
    return 1/(1 - beta**2)**0.5


def momentum(beta:float, gamma:float) -> float:
    """
    Compute central reference momentum

    """
    return beta*gamma


def rigidity(beta:float, gamma:float, mass:float, ratio:int=1) -> float:
    """
    Compute magnetic rigidity (Tm)

    """
    return 1E6/CL*momentum(beta, gamma)*mass*ratio


def ptc(qsps:Array,
        kind:str,
        parameters:dict[str, str|int|float], *,
        gamma:float=10.0**9,
        exact:bool=True,
        tx:float=0.0,
        ty:float=0.0,
        tz:float=0.0,
        rx:float=0.0,
        ry:float=0.0,
        rz:float=0.0) -> Array:
    """


    Parameters
    ----------
    qsps: Array
        initial condition (sector ordering)
    kind: sts
        element kind (drift, quadrupole, ...)
    parameters: dict[str, str|int|float]
        element parameters
    gamma: float, default=10**9
        gamma
    exact: bool, defaul=True
        exact hamiltonian and alignment flag
    tx: float, default=0.0
        tx
    ty: float, default=0.0
        ty
    tz: float, default=0.0
        tz
    rx: float, default=0.0
        rx
    ry: float, default=0.0
        ry
    rz: float, default=0.0
        rz

    Returns
    -------
    Array

    """
    q_x, q_y, q_s, p_x, p_y, p_s = list(qsps)
    file = Path('ptc')
    data = Path('track.obs0001.p0001')
    code = f"""
    mag:{kind},{''.join([f'{key}={str(value)}, ' for key, value in parameters.items()])};
    map:line=(mag) ;
    beam,gamma={gamma},particle=electron ;
    set,format="20.20f","-20s" ;
    use,period=map ;
    select,flag=error,pattern="mag" ;
    select,flag=error,pattern="mag" ;
    ealign,dx={tx},dy={ty},ds={tz},dphi={rx},dtheta={ry},dpsi={rz} ;
    ptc_create_universe,sector_nmul_max=10,sector_nmul=10 ;
    ptc_create_layout,model=1,method=6,nst=1000,exact={str(exact).lower()} ;
    ptc_setswitch,fringe=false,time=true,totalpath=false,exact_mis={str(exact).lower()} ;
    ptc_align ;
    ptc_start,x={q_x},px={p_x},y={q_y},py={p_y},t={q_s},pt={p_s} ;
    ptc_track,icase=6,turns=1,file=track,maxaper={{1.,1.,1.,1.,1.,1.}} ;
    ptc_track_end ;
    ptc_end ;

    """
    with file.open('w', encoding='utf-8') as stream:
        stream.write(code)
    system(f'madx < {str(file)} > /dev/null')
    with data.open('r', encoding='utf-8') as stream:
        line = ''
        for line in stream:
            continue
        _, _, q_x, p_x, q_y, p_y, q_s, p_s, *_ = line.split()
    file.unlink()
    data.unlink()
    return jax.numpy.asarray([float(x) for x in (q_x, q_y, q_s, p_x, p_y, p_s)])


def bessel(x:Array, n:int=0) -> Array:
    """
    Bessel function (series approximation for small argument upto order 10)

    """
    if n == 0:
        return 1 - x**2/4 + x**4/64 - x**6/2304 + x**8/147456 - x**10/14745600
    return x**n/2**n*(
        1/factorial(n) -
        1/factorial(n + 1)*x**2/4 +
        1/factorial(n + 2)*x**4/32 -
        1/factorial(n + 3)*x**6/384 +
        1/factorial(n + 4)*x**8/6144 -
        1/factorial(n + 5)*x**10/122880
    )
