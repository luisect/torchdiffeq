from .dopri5 import Dopri5Solver
from .bosh3 import Bosh3Solver
from .adaptive_heun import AdaptiveHeunSolver
from .fixed_grid import Euler, Midpoint, RK4
from .fixed_adams import AdamsBashforth, AdamsBashforthMoulton
from .dopri8 import Dopri8Solver
from .scipy_wrapper import ScipyWrapperODESolver
from .misc import _check_inputs, _flat_to_shape

SOLVERS = {
    'dopri8': Dopri8Solver,
    'dopri5': Dopri5Solver,
    'bosh3': Bosh3Solver,
    'adaptive_heun': AdaptiveHeunSolver,
    'euler': Euler,
    'midpoint': Midpoint,
    'rk4': RK4,
    'explicit_adams': AdamsBashforth,
    'implicit_adams': AdamsBashforthMoulton,
    # Backward compatibility: use the same name as before
    'fixed_adams': AdamsBashforthMoulton,
    # ~Backwards compatibility
    'scipy_solver': ScipyWrapperODESolver,
}


def odeint(func, y0, t, rtol=1e-7, atol=1e-9, method=None, options=None):
   
    shapes, func, y0, t, rtol, atol, method, options = _check_inputs(func, y0, t, rtol, atol, method, options, SOLVERS)

    solver = SOLVERS[method](func=func, y0=y0, rtol=rtol, atol=atol, **options)
    solution = solver.integrate(t)
        
    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)
    return solution 

 
