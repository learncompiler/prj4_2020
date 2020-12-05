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

import torch
import torchvision
from torchvision import datasets, transforms

from PIL import Image
# import matplotlib.pyplot as plt
import time


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=300, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=100, shuffle=True)



# transform = transforms.Compose(
#     [transforms.ToTensor(), 
#     transforms.Normalize((0.1307,), (0.3081,))])

# data_test = datasets.MNIST(
#   root='data',
#   train=False,
# )
# for i in range(50, 500):
#   if data_test.__getitem__(i)[1] == 9:
#     img = data_test.__getitem__(i)[0]
#     plt.figure("number")
#     plt.imshow(img)
#     plt.show()
#     break

batch_size = 300

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

def inferece_mlp(data_shape):
    data = relay.var("data", shape=data_shape, dtype="float32")
    fc1 = relay.nn.dense(data, relay.var("fc1_weight"), units=128)
    fc1 = relay.nn.bias_add(fc1, relay.var("fc1_bias"), axis=-1)
    act1 = relay.nn.relu(fc1)
    fc2 = relay.nn.dense(act1, relay.var("fc2_weight"), units=64)
    fc2 = relay.nn.bias_add(fc2, relay.var("fc2_bias"), axis=-1)
    act2 = relay.nn.relu(fc2)
    fc3 = relay.nn.dense(act2, relay.var("fc3_weight"), units=10)
    fc3 = relay.nn.bias_add(fc3, relay.var("fc3_bias"), axis=-1)
    mlp = relay.nn.softmax(data=fc3)
    args = relay.analysis.free_vars(mlp)
    return relay.Function(args, mlp)

def mlp_training(epoch=10):
    data_shape = (batch_size, 784)
    label_shape = (batch_size, 10)
    valid_shape = (100, 784)
    dtype = "float32"
    func = normal_mlp(data_shape, label_shape)
    # inference_t = inferece_mlp(data_shape)
    # inference = inferece_mlp(valid_shape)
    # inference_t = run_infer_type(inference_t)
    # inference = run_infer_type(inference)
    func = run_infer_type(func)
    back = run_infer_type(gradient(func, mode="higher_order"))
    # seq1 = tvm.transform.Sequential([PartialEvaluate(), DeadCodeElimination()])
    # seq1 = tvm.transform.Sequential([DeadCodeElimination()])
    # back = run_infer_type(seq1(back))
    # ex = create_executor(target="cuda")

    mod, params = create_workload(back)
    # mod = seq1(mod)
    print(params)
    w1, w1b, w2, w2b, w3, w3b = (i.asnumpy() for i in params.values())
    
    opt_level = 0
    # target = tvm.target.cuda()
    with tvm.transform.PassContext(opt_level=opt_level, disabled_pass=['PartialEvaluate']):
        lib = relay.build(mod, target='llvm', params=params)


    # loss = 0 
    # cnt = 0
    # lr = 0.01
    # cost = 0

    # for i in range(0, epoch):
    #     for batch_idx, (data, target) in enumerate(train_loader):
    #         cnt += 1
                
    #         data = data.view(-1, 28*28)
    #         data = np.array(data)
    #         label = np.array(np.eye(10, dtype="float32")[target])

            
    #         # forward, grad = ex.evaluate(back)(data, w1, w1b, w2, w2b, w3, w3b, label)    
    #         t1 = time.time()
    #         w1 = w1 - lr * grad[1].asnumpy()
    #         w1b = w1b - lr * grad[2].asnumpy()
    #         w2 = w2 - lr * grad[3].asnumpy()
    #         w2b = w2b - lr * grad[4].asnumpy()
    #         w3 = w3 - lr * grad[5].asnumpy()
    #         w3b = w3b - lr * grad[6].asnumpy()
    #         cost += time.time() - t1
    #         loss = loss + forward.asnumpy()

    #         if cnt % 100 == 0:
    #             print(cnt, ": ", loss / cnt)

    #     print("epoch %d: %f" % (i, loss / cnt))
    #     print("time: %f" % (cost))
    #     loss = 0
    #     cnt = 0
        # equal = 0
        # for batch_idx, (data, target) in enumerate(train_loader):
        #     data = data.view(-1, 28*28)
        #     data = np.array(data)
        #     forward = ex.evaluate(inference_t)(data, w1, w1b, w2, w2b, w3, w3b)
        #     equal += np.sum(forward.asnumpy().argmax(1) == np.array(target))
        # print("train acc: %f" % (equal / 60000))

        # equal = 0
        # for batch_idx, (data, target) in enumerate(test_loader):
        #     data = data.view(-1, 28*28)
        #     data = np.array(data)
        #     forward = ex.evaluate(inference)(data, w1, w1b, w2, w2b, w3, w3b)
        #     equal += np.sum(forward.asnumpy().argmax(1) == np.array(target))
        # print("test acc: %f" % (equal / 10000))

if __name__ == "__main__":
    mlp_training()