from tvm import relay
import tvm.relay.op as op
from tvm.relay.testing import run_infer_type, rand, create_workload
from tvm.relay.transform import gradient
from tvm.relay import create_executor
import numpy as np
import tvm
from trainable_model import Trainable_model
import time


def test_mlp():
    shape = (3, 3)
    dtype = 'float32'
    weight_shape = (1, 3)
    y_shape = (3, 1)
    t = relay.TensorType(shape, dtype)
    weight_t = relay.TensorType(weight_shape, dtype)
    y_t = relay.TensorType(y_shape, dtype)
    x = relay.var("x", t)
    weight = relay.var("weight", weight_t)
    y = relay.var("y", y_t)

    mlp = relay.Function(
        [x, weight, y], (y-op.nn.dense(x, weight))*(y-op.nn.dense(x, weight)))
    mlp = run_infer_type(mlp)

    back_func = run_infer_type(gradient(mlp))

    ex = create_executor()
    input_x = tvm.nd.array((-1+2*np.random.rand(*shape)).astype(dtype))
    input_weight = tvm.nd.array(np.array([[1.5, 2.0, 2.0]], dtype=dtype))
    input_y = tvm.nd.array(np.array([[1.0], [0.0], [0.0]], dtype=dtype))
    # ret = ex.evaluate(func)(input_data)
    # print(input_data)
    # print(ret)
    # print(back_func)
    print(back_func.checked_type)
    back_func_run = ex.evaluate(back_func)
    print(type(back_func_run))
    forward, (grad1, grad2, grad3) = back_func_run(
        input_x, input_weight, input_y)
    print(forward)
    print(grad1)
    print(grad2)
    print(grad3)

    # print(forward)
    # print(grad)

    # print(type(func), func)
    # func = run_infer_type(func)
    # print(type(func), func)
    # backed_func = run_infer_type(gradient(func))
    # # print(backed_func)
    # print(backed_func.checked_type)
    # assert backed_func.checked_type == relay.FuncType([t], relay.TupleType([t, relay.TupleType([t])]))


def test_mnist_cnn():
    model = Trainable_model(lr=0.01)

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
    label = relay.var("data", shape=label_shape, dtype="float32")

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
    print(args)
    model.create(args, mlp)

    model.init_param_values()

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

            loss += model.run(data, label)

            if cnt % 100 == 0:
                print(cnt, ": ", loss / cnt)
        end = time.time()
        sum_time += end - start
        print("one epoch time:", end-start)
        print("epoch %d: %f" % (i, loss / cnt))
        loss = 0
        cnt = 0
        
    print('average epoch time:',sum_time / epoch)

def test_mnist():
    model = Trainable_model(lr=0.01)

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

    data_shape = (batch_size, 784)
    label_shape = (batch_size, 10)

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
    print(args)
    model.create(args, mlp)

    model.init_param_values()

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

            loss += model.run(data, label)

            if cnt % 100 == 0:
                print(cnt, ": ", loss / cnt)

        print("epoch %d: %f" % (i, loss / cnt))
        end = time.time()
        sum_time += end - start
        print("one epoch time:", end-start)
        loss = 0
        cnt = 0
    
    print('average epoch time:',sum_time / epoch)

    # exit()
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

if __name__ == "__main__":
    # test_mlp()
    # a = [0, 1, 2]
    # print(*a)
    # test_mnist()
    test_mnist_cnn()
    # batch_size = 300
    # data_shape = (batch_size, 1, 28, 28)
    # # relay.TensorType(data_shape)
    # data = relay.var("data", relay.TensorType(data_shape, "float32"))
    # # label = relay.var("data", shape=label_shape, dtype="float32")

    # fc1 = relay.nn.conv2d(data, relay.var("fc1_weight"), kernel_size=(5, 5), channels=16)
    # run_infer_type(fc1)
