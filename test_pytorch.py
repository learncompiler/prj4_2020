import torch
import time
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms


def test_mnist():
    batch_size = 300

    def get_dataloader(batch_size):
        # import torch
        # from torchvision import datasets, transforms
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

    model = torch.nn.Sequential(nn.Linear(784, 128),
                                nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10),
                                nn.LogSoftmax())

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()
    loss = 0
    cnt = 0
    epoch = 3
    sum_time = 0
    lr = 0.01
    for i in range(0, epoch):
        start = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            cnt += 1

            data = data.view(-1, 28*28)
            # print(target.size())
            # label = torch.eye(10)[target]
            output = model(data)
            # print(output.size())
            # print(label.size())

            temp_loss = criterion(output, target)
            temp_loss.backward()
            for f in model.parameters():
                f.data.sub_(f.grad.data * lr)
            # optimizer.step()
            # optimizer.zero_grad()
            model.zero_grad()
            loss += temp_loss.item()


            if cnt % 100 == 0:
                print(cnt, ": ", loss / cnt)

        print("epoch %d: %f" % (i, loss / cnt))
        end = time.time()
        sum_time += end - start
        print("one epoch time:", end-start)
        print("cnt:", cnt)
        loss = 0
        cnt = 0
    print('average epoch time:',sum_time / epoch)


def test_mnist_cnn():
    batch_size = 300

    def get_dataloader(batch_size):
        # import torch
        # from torchvision import datasets, transforms
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

    model = torch.nn.Sequential(nn.Conv2d(1, 16, kernel_size=(5, 5)),
                                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), nn.Conv2d(16, 16, kernel_size=(5, 5)),
                                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), nn.Flatten(start_dim=1),
                                nn.Linear(256, 10),
                                nn.LogSoftmax())

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()
    loss = 0
    cnt = 0
    epoch = 3
    sum_time = 0
    lr = 0.01
    for i in range(0, epoch):
        start = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            cnt += 1

            data = data.view(-1, 1, 28, 28)
            # print(target.size())
            # label = torch.eye(10)[target]
            output = model(data)
            # print(output.size())
            # print(label.size())

            temp_loss = criterion(output, target)
            temp_loss.backward()
            for f in model.parameters():
                f.data.sub_(f.grad.data * lr)
            # optimizer.step()
            # optimizer.zero_grad()
            model.zero_grad()
            loss += temp_loss.item()


            if cnt % 100 == 0:
                print(cnt, ": ", loss / cnt)

        print("epoch %d: %f" % (i, loss / cnt))
        end = time.time()
        sum_time += end - start
        print("one epoch time:", end-start)
        print("cnt:", cnt)
        loss = 0
        cnt = 0
    print('average epoch time:',sum_time / epoch)

if __name__ == '__main__':
    # test_mnist()
    test_mnist_cnn()