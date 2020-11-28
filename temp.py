from numpy.core import function_base

import tvm

from tvm import relay

from tvm.relay import GlobalVar

from tvm.relay.op.nn.nn import batch_flatten

from tvm.relay.op.tensor import cos

from tvm.relay.testing import run_infer_type

from tvm.relay import create_executor, transform

from tvm.relay.transform import gradient, PartialEvaluate, DeadCodeElimination

from tvm.relay.testing import run_infer_type, rand, check_grad, create_workload

import numpy as np

from tvm.relay.transform.transform import ForwardFoldScaleAxis

from tvm.contrib import graph_runtime as runtime

def normal_mlp(data_shape, label_shape):
    data = relay.var("data", shape=data_shape, dtype="float32")
    label = relay.var("data", shape=label_shape, dtype="float32")
    fc1 = relay.nn.dense(data, relay.var("fc1_weight"), units=128)
    fc1 = relay.nn.bias_add(fc1, relay.var("fc1_bias"), axis=-1)
    act1 = relay.nn.relu(fc1)
    fc2 = relay.nn.dense(act1, relay.var("fc2_weight"), units=64)
    fc2 = relay.nn.bias_add(fc2, relay.var("fc2_bias"), axis=-1)
    act2 = relay.nn.relu(fc2)
    fc3 = relay.nn.dense(act2, relay.var("fc3_weight"), units=10)
    fc3 = relay.nn.bias_add(fc3, relay.var("fc3_bias"), axis=-1)
    mlp = relay.nn.softmax(data=fc3)
    mlp = relay.nn.cross_entropy(mlp, label)
    args = relay.analysis.free_vars(mlp)
    return relay.Function(args, mlp)

batch_size = 10

data_shape = (batch_size, 784)

label_shape = (batch_size, 10)

valid_shape = (100, 784)

dtype = "float32"

func = normal_mlp(data_shape, label_shape)

func = run_infer_type(func)

back = run_infer_type(gradient(func))

mod, params = create_workload(back)

opt_level = 0

with tvm.transform.PassContext(opt_level=opt_level):

    lib = relay.build(mod, target='llvm', params=params)