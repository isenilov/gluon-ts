# Third-party imports
import mxnet as mx
import numpy as np
import scipy.special

# TODO: Remove these once
#
#    https://github.com/apache/incubator-mxnet/pull/13229
#    https://github.com/apache/incubator-mxnet/pull/13811
#
# are released.


class Erf(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0].asnumpy()
        y = scipy.special.erf(x)
        self.assign(out_data[0], req[0], mx.nd.array(y))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        grad = (
            2.0 / np.sqrt(np.pi) * (-in_data[0]).square().exp() * out_grad[0]
        )
        self.assign(in_grad[0], req[0], grad)


@mx.operator.register("erf")
class ErfProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(ErfProp, self).__init__(True)

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return Erf()


class ErfInv(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0].asnumpy()
        y = scipy.special.erfinv(x)
        self.assign(out_data[0], req[0], mx.nd.array(y))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        grad = 0.5 * np.sqrt(np.pi) * out_data[0].square().exp() * out_grad[0]
        self.assign(in_grad[0], req[0], grad)


@mx.operator.register("erfinv")
class ErfInvProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(ErfInvProp, self).__init__(True)

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return ErfInv()
