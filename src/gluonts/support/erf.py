import mxnet as mx

from gluonts.model.common import Tensor

MXNET_HAS_ERF = hasattr(mx.nd, 'erf')


def erf(F, x: Tensor):
    if MXNET_HAS_ERF:
        return F.erf(x)
    # Using numerical recipes approximation for erf function
    # accurate to 1E-7

    ones = x.ones_like()
    zeros = x.zeros_like()
    t = ones / (ones + 0.5 * x.abs())

    coefficients = [
        1.00002368,
        0.37409196,
        0.09678418,
        -0.18628806,
        0.27886807,
        -1.13520398,
        1.48851587,
        -0.82215223,
        0.17087277,
    ]

    inner = zeros
    for c in coefficients[::-1]:
        inner = t * (c + inner)

    res = ones - t * (inner - 1.26551223 - x.square()).exp()
    return F.where(F.broadcast_greater_equal(x, zeros), res, -1.0 * res)
