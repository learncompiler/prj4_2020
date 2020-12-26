from numpy.core import function_base
import tvm
from tvm import relay
from tvm.relay import GlobalVar
import tvm.relay.op as op
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
from trainable_model import Trainable_model

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

def firstorder_mlp():
    shape = (3, 3)
    dtype = 'float32'
    weight_shape = (1, 3)
    y_shape = (3, 3)
    t = relay.TensorType(shape, dtype)
    weight_t = relay.TensorType(weight_shape, dtype)
    y_t = relay.TensorType(y_shape, dtype)
    x = relay.var("data", t)
    weight = relay.var("weight", weight_t)
    y = relay.var("y", y_t)

    # mlp = relay.Function(
    #     [x, weight, y], (y-op.nn.dense(x, weight))*(y-op.nn.dense(x, weight)))
    mlp = relay.Function([x, y], x + x + y)
    print(mlp)
    mlp = run_infer_type(mlp)
    back_func = run_infer_type(gradient(mlp, mode="first_order"))
    print(back_func)
    mod, params = create_workload(back_func)
    # mod, params = create_workload(mlp)
    print(mod)
    opt_level = 0
    with tvm.transform.PassContext(opt_level=opt_level):
        lib = relay.build(mod, target='llvm', params=params)

    ctx = tvm.cpu(0)
    module = runtime.GraphModule(lib['default'](ctx))
    input_x = tvm.nd.array((-1+2*np.random.rand(*shape)).astype(dtype))
    input_y = tvm.nd.array((-1+2*np.random.rand(*shape)).astype(dtype))
    params['data'] = input_x
    params['y'] = input_y

    module.set_input(**params)
    # module.set_input('y', input_y)
    module.run()
    output = module.get_output(0)
    gradx = module.get_output(1)
    grady = module.get_output(2)
    print(output)
    print(gradx)
    print(grady)
    # ex = create_executor()
    # back_func_run = ex.evaluate(back_func)
    # input_x = tvm.nd.array((-1+2*np.random.rand(*shape)).astype(dtype))
    # forward, (grad1, ) = back_func_run(input_x)
    # print(type(forward))
    # print(type(grad1))
    # print(forward)
    # print(grad1)

def normal_mlp(data_shape, label_shape):
    data = relay.var("data", shape=data_shape, dtype="float32")
    label = relay.var("y", shape=label_shape, dtype="float32")
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

def mlp_training():
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
    back = run_infer_type(gradient(func, mode="first_order"))
    # seq1 = tvm.transform.Sequential([PartialEvaluate(), DeadCodeElimination()])
    # seq1 = tvm.transform.Sequential([DeadCodeElimination()])
    # back = run_infer_type(seq1(back))
    # ex = create_executor(target="cuda")

    mod, params = create_workload(back)
    # mod = seq1(mod)
    print(params.keys())
    # w1, w1b, w2, w2b, w3, w3b = (i.asnumpy() for i in params.values())
    # exit()
    opt_level = 3
    # target = tvm.target.cuda()
    with tvm.transform.PassContext(opt_level=opt_level, disabled_pass=['FoldScaleAxis']):
        lib = relay.build(mod, target='llvm')

    ctx = tvm.cpu(0)
    module = runtime.GraphModule(lib['default'](ctx))
    model = Trainable_model(lr=0.01)
    model.create_from_graph_runtime_module(module)
    model.init_param_values(**params)
    
    loss = 0
    cnt = 0
    epoch = 3
    sum_time = 0
    for i in range(0, epoch):
        start = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            cnt += 1
                
            data = data.view(-1, 28*28)
            data = np.array(data)
            label = np.array(np.eye(10, dtype="float32")[target])
            data = tvm.nd.array(data)
            label = tvm.nd.array(label)

            loss += model.build_run(data=data,y=label)

            if cnt % 100 == 0:
                print(cnt, ": ", loss / cnt)

        print("epoch %d: %f" % (i, loss / cnt))
        end = time.time()
        sum_time += end - start
        print("one epoch time:", end-start)
        loss = 0
        cnt = 0
    print('average epoch time:',sum_time / epoch)

    # for batch_idx, (data, target) in enumerate(test_loader):
    #     cnt += 1
            
    #     data = data.view(-1, 28*28)
    #     data = np.array(data)
    #     label = np.array(np.eye(10, dtype="float32")[target])

    #     loss += model.run(data, label, back=False)

    #     if cnt % 100 == 0:
    #         print(cnt, ": ", loss / cnt)

    #     print("test loss : %f" % (loss / cnt))
    #     loss = 0
    #     cnt = 0


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

def cnn_training():
    # model = Trainable_model(lr=0.01)

    batch_size = 300

    def get_dataloader(batch_size):
        import torch
        from torchvision import datasets, transforms
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True)


        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=batch_size, shuffle=True)

        return train_loader, test_loader

    train_loader, test_loader = get_dataloader(batch_size)

    data_shape = (batch_size, 1, 28, 28)
    label_shape = (batch_size, 10)

    data = relay.var("data", shape=data_shape, dtype="float32")
    label = relay.var("y", shape=label_shape, dtype="float32")

    fc1 = relay.nn.conv2d(data, relay.var("fc1_weight"), kernel_size=(5, 5), channels=16)

    # fc1 = relay.nn.dense(data, relay.var("fc1_weight"), units=128)
    # fc1 = relay.nn.bias_add(fc1, relay.var("fc1_bias"), axis=-1)
    # act1 = relay.nn.relu(fc1)

    fc2 = relay.nn.max_pool2d(fc1, pool_size=(2, 2), strides=(2, 2))

    # fc2 = relay.nn.dense(act1, relay.var("fc2_weight"), units=64)
    # fc2 = relay.nn.bias_add(fc2, relay.var("fc2_bias"), axis=-1)
    # act2 = relay.nn.relu(fc2)

    fc3 = relay.nn.conv2d(fc2, relay.var("fc3_weight"), kernel_size=(5, 5), channels=16)

    # fc3 = relay.nn.dense(act2, relay.var("fc3_weight"), units=10)
    # fc3 = relay.nn.bias_add(fc3, relay.var("fc3_bias"), axis=-1)
    
    fc4 = relay.nn.max_pool2d(fc3, pool_size=(2, 2), strides=(2, 2))

    fc5 = relay.nn.batch_flatten(fc4)

    fc6 = relay.nn.dense(fc5, relay.var("fc6_weight"), units=10)
    fc6 = relay.nn.bias_add(fc6, relay.var("fc6_bias"), axis=-1)

    mlp = relay.nn.softmax(data=fc6)
    mlp = relay.nn.cross_entropy(mlp, label)

    args = relay.analysis.free_vars(mlp)
    func = relay.Function(args, mlp)

    func = run_infer_type(func)
    back = run_infer_type(gradient(func, mode="first_order"))

    mod, params = create_workload(back)

    opt_level = 3
    # target = tvm.target.cuda()
    with tvm.transform.PassContext(opt_level=opt_level, disabled_pass=['FoldScaleAxis']):
        lib = relay.build(mod, target='llvm')

    # print(args)
    ctx = tvm.cpu(0)
    module = runtime.GraphModule(lib['default'](ctx))
    model = Trainable_model(lr=0.01)
    model.create_from_graph_runtime_module(module)
    model.init_param_values(**params)

    loss = 0
    cnt = 0
    epoch = 3
    sum_time = 0
    for i in range(0, epoch):
        start = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            cnt += 1
                
            # data = data.view(-1, 28*28)
            # print('hello')
            data = np.array(data)
            # data = data[:, np.newaxis, :, :]
            # print(data.shape)
            label = np.array(np.eye(10, dtype="float32")[target])
            # data = tvm.nd.array(data)
            # label = tvm.nd.array(label)

            loss += model.build_run(data=data, y=label)

            if cnt % 100 == 0:
                print(cnt, ": ", loss / cnt)
        end = time.time()
        sum_time += end - start
        print("one epoch time:", end-start)
        print("epoch %d: %f" % (i, loss / cnt))
        loss = 0
        cnt = 0
        
    print('average epoch time:',sum_time / epoch)

if __name__ == "__main__":
    mlp_training()
    # firstorder_mlp()
    # cnn_training()