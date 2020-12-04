import torch
import torch.nn as nn
from .odeint import SOLVERS, odeint
from .misc import _check_inputs, _flat_to_shape, _rms_norm, _mixed_linf_rms_norm, _wrap_norm


class OdeintAdjointMethod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, shapes, func, y0, t, rtol, atol, method, options, adjoint_rtol, adjoint_atol, adjoint_method,
                adjoint_options, t_requires_grad, *adjoint_params):

        ctx.shapes = shapes
        ctx.func = func
        ctx.adjoint_rtol = adjoint_rtol
        ctx.adjoint_atol = adjoint_atol
        ctx.adjoint_method = adjoint_method
        ctx.adjoint_options = adjoint_options
        ctx.t_requires_grad = t_requires_grad

        print("Forward adjoint")
        with torch.no_grad():
            y = odeint(func, y0, t, rtol=rtol, atol=atol, method=method, options=options)
        ctx.save_for_backward(t, y, *adjoint_params)
        return y

##    
   
    
   
##

def odeint_adjoint(func, y0, t, rtol=1e-7, atol=1e-9, method=None, options=None, adjoint_rtol=None, adjoint_atol=None,
                   adjoint_method=None, adjoint_options=None, adjoint_params=None):

    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if adjoint_params is None and not isinstance(func, nn.Module):
        raise ValueError('func must be an instance of nn.Module to specify the adjoint parameters; alternatively they '
                         'can be specified explicitly via the `adjoint_params` argument. If there are no parameters '
                         'then it is allowable to set `adjoint_params=()`.')

    # Must come before _check_inputs as we don't want to use normalised input (in particular any changes to options)
    if adjoint_rtol is None:
        adjoint_rtol = rtol
    if adjoint_atol is None:
        adjoint_atol = atol
    if adjoint_method is None:
        adjoint_method = method
    if adjoint_options is None:
        adjoint_options = {k: v for k, v in options.items() if k != "norm"} if options is not None else {}
    if adjoint_params is None:
        adjoint_params = tuple(find_parameters(func))
    else:
        adjoint_params = tuple(adjoint_params)  # in case adjoint_params is a generator.

    # Filter params that don't require gradients.
    adjoint_params = tuple(p for p in adjoint_params if p.requires_grad)

    # Normalise to non-tupled input
    shapes, func, y0, t, rtol, atol, method, options = _check_inputs(func, y0, t, rtol, atol, method, options, SOLVERS)

    if "norm" in options and "norm" not in adjoint_options:
        adjoint_shapes = [torch.Size(()), y0.shape, y0.shape] + [torch.Size([sum(param.numel() for param in adjoint_params)])]
        adjoint_options["norm"] = _wrap_norm([_rms_norm, options["norm"], options["norm"]], adjoint_shapes)

    solution = OdeintAdjointMethod.apply(shapes, func, y0, t, rtol, atol, method, options, adjoint_rtol, adjoint_atol,
                                         adjoint_method, adjoint_options, t.requires_grad, *adjoint_params)

    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)
    return solution


def find_parameters(module):

    assert isinstance(module, nn.Module)

    if getattr(module, '_is_replica', False):

        def find_tensor_attributes(module):
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v) and v.requires_grad]
            return tuples

        gen = module._named_members(get_members_fn=find_tensor_attributes)
        return [param for _, param in gen]
    else:
        return list(module.parameters())
