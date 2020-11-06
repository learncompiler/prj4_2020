import tvm
from tvm import relay
from tvm.relay import nn, create_executor
from tvm.relay.testing import run_infer_type, create_workload
from tvm.relay.transform import gradient


class Trainable_model:

    def __init__(self, lr=0.01):
        # self.params = []
        self.lr = lr

    def create(self, args, body):
        # self.input = args[0]
        # self.params = args[1:-1]
        # self.output = args[-1]
        self.forward_func = relay.Function(args, body)
        self.forward_func = run_infer_type(self.forward_func)
        self.backward_func = run_infer_type(gradient(self.forward_func))
        self.only_forward_model = create_executor().evaluate(self.forward_func)
        self.model = create_executor().evaluate(self.backward_func)

    def init_param_values(self, *values):
        if values:
            # print('have values')
            self.param_values = list(values)
        else:
            _, params = create_workload(self.forward_func)
            self.param_values = [i.asnumpy() for i in params.values()]

    def run(self, *data, back=True):
        if back:
            forward, grad = self.model(data[0], *self.param_values, data[-1])
            for idx in range(len(grad)-2):
                self.param_values[idx] -= self.lr * grad[idx+1].asnumpy()
        else:
            forward = self.only_forward_model(data[0], *self.param_values, data[-1])
        
        return forward.asnumpy() # return loss
