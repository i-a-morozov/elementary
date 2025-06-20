"""
Util
----

Utils module

"""
from os import system
from pathlib import Path

import jax
from jax import Array


def ptc(qsps:Array,
        element:str, *,
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
    element: str
        input element (type, ..., parameter=value, ...)
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
    mag:{element};
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
