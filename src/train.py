import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from component.blueprint.data_generator import GridWorldData
from utils.utils import print_header
from utils.utils import print_stats
from utils.utils import get_stats
from model.vin import VIN


def train_model(net, trainloader, args, criterion, optimizer):
    print_header()

    # automatically select device to make the code device agnostic
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    # loop over epochs
    for epoch in range(args.epochs):
        avg_error, avg_loss, num_batches = 0.0, 0.0, 0.0
        start_time = time.time()

        # loop over batches of data
        for i, data in enumerate(trainloader):
            # get input data
            X, S1, S2, labels = [d.to(device) for d in data]
            if X.size()[0] != args.batch_size:
                continue  # drop those data if not enough for a batch

            # implement backpropagation and update params
            optimizer.zero_grad()
            outputs, predictions = net(X, S1, S2, args.num_vi, visualize=False)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # get stats
            loss_batch, error_batch = get_stats(loss, predictions, labels)
            avg_loss += loss_batch
            avg_error += error_batch
            num_batches += 1
        time_duration = time.time() - start_time

        # print epoch logs
        print_stats(epoch, avg_loss, avg_error, num_batches, time_duration)

    print('\nFinished training. \n')


def test_model(net, testloader, args):
    total, correct = 0.0, 0.0

    # Automatically select device, device agnostic
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    for i, data in enumerate(testloader):
        # Get inputs
        X, S1, S2, labels = [d.to(device) for d in data]
        if X.size()[0] != args.batch_size:
            continue

        # Forward pass
        outputs, predictions = net(X, S1, S2, args.num_vi, visualize=False)

        # Select actions with max scores(logits)
        _, predicted = torch.max(predictions, dim=1, keepdim=True)

        # Unwrap autograd.Variable to Tensor
        predicted = predicted.data

        # Compute test accuracy
        correct += (torch.eq(torch.squeeze(predicted), labels)).sum()
        total += labels.size()[0]
    print('Test Accuracy: {:.2f}%'.format(100 * (correct / total)))


def set_args():
    # Parsing training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', type=str,
                        default='../data/gridworld_8x8.npz', help='Path to data file')
    parser.add_argument('--dom_size', type=int,
                        default=8, help='Size of image')
    parser.add_argument('--lr', type=float,
                        default=0.002, help='learning rate, [0.01, 0.005, 0.002, 0.001]')
    parser.add_argument('--epochs', type=int,
                        default=30, help='number of epochs to train')
    parser.add_argument('--num_vi', type=int,
                        default=10, help='number of Value Iterations')
    parser.add_argument('--num_input', type=int,
                        default=2, help='number of channels in input layer')
    parser.add_argument('--num_hidden', type=int,
                        default=150, help='number of channels in first hidden layer')
    parser.add_argument('--num_qlayer', type=int,
                        default=10, help='number of channels in q layer (~actions) in VI-module')
    parser.add_argument('--batch_size', type=int,
                        default=128, help='batch size')
    return parser.parse_args()


if __name__ == '__main__':
    # set args
    args = set_args()

    # extract dataset
    transform = None
    trainset = GridWorldData(args.datafile, dom_size=args.dom_size,
                             train=True, transform=transform)
    testset = GridWorldData(args.datafile, dom_size=args.dom_size,
                            train=False, transform=transform)

    # make dataloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=0)

    # instantiate VIN model
    vin = VIN(args)

    # set loss func
    criterion = nn.CrossEntropyLoss()

    # set optimizer
    optimizer = optim.RMSprop(vin.parameters(), lr=args.lr, eps=1e-6)

    # train model
    train_model(net=vin, trainloader=trainloader, args=args,
                criterion=criterion, optimizer=optimizer)

    # test model
    test_model(net=vin, testloader=testloader, args=args)

    # save the trained model params
    save_path = '../data/vin_{0}x{0}.pth'.format(args.dom_size)
    torch.save(vin.state_dict(), save_path)
