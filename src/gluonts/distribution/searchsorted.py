# Third-party imports
import mxnet as mx
import numpy as np


class SearchSorted(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        sorted_values = in_data[0].asnumpy()
        x = in_data[1].asnumpy()
        batch_size, num_timesteps, target_dim = x.shape
        index_tensor = np.zeros(x.shape)

        for batch in range(batch_size):
            for dimension in range(target_dim):
                ts = x[batch, :, dimension]
                sorted_past_ts = sorted_values[batch, :, dimension]
                indices_left = np.searchsorted(sorted_past_ts, ts, side='left')
                indices_right = np.searchsorted(
                    sorted_past_ts, ts, side='right'
                )
                indices = indices_left + (indices_right - indices_left) // 2
                indices = indices - 1
                indices = np.minimum(indices, len(sorted_past_ts) - 1)
                indices[indices < 0] = 0
                index_tensor[batch, :, dimension] = indices

        self.assign(out_data[0], req[0], mx.nd.array(index_tensor))

    # TODO: Check whether this needs to have a backwards method implemented
    # def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
    #    grad = 2.0 / np.sqrt(np.pi) * (-in_data[0]).square().exp() * out_grad[0]
    #    self.assign(in_grad[0], req[0], grad)


@mx.operator.register("searchsorted")
class SearchSortedProb(mx.operator.CustomOpProp):
    def __init__(self):
        super(SearchSortedProb, self).__init__(True)

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return SearchSorted()

    def list_arguments(self):
        return ['sorted', 'data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        sorted_shape = in_shape[0]
        data_shape = in_shape[1]
        output_shape = in_shape[1]
        return [sorted_shape, data_shape], [output_shape], []
