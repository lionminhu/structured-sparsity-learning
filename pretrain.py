import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import pdb
from os import getcwd

from models import CustomNet
from params import Params
from dataloader import prep_dataloaders


def scale_lr(optimizer, scale):
    """Multiplies the optimizer's learning rate by `scale`"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= scale


def train(model, optimizer, criterion, trainloader):
    """A single training iteration"""
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    # Batch loop
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if params.use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # A single optimization step
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Update loss and accuracy info
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print("Train: Loss: %.3f, Acc: %.3f (%d/%d)" % (train_loss / \
          (batch_idx + 1), correct / total * 100., correct, total))
    
    return correct / total


def test(model, optimizer, criterion, testloader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    # Batch loop
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if params.use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()
        
        # A single test iteration
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()

        # Update loss and accuracy info
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    print("Test: Loss: %.3f, Acc: %.3f (%d/%d)" % (test_loss / \
          (batch_idx + 1), correct / total * 100., correct, total))

    return correct / total


if __name__ == "__main__":
    params = Params()

    # Get dataset
    trainloader, testloader, classes = prep_dataloaders()

    # Set up model, optimizer and loss
    model = CustomNet(len(classes), use_gpu=params.use_gpu)
    # optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9,
    #                             weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, 
                                 weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # Use GPU if given
    if params.use_gpu:
        model = model.cuda()

    best_train_acc = 0
    best_test_acc = 0

    # Loop over training epochs
    for epoch in range(params.num_pretrain_epochs):
        print("========== epoch %d" % (epoch))

        # Update learning rate of optimizers regularly
        if epoch % 50 == 0:
            scale_lr(optimizer, 0.1)

        # Train
        tic = time.time()
        train_acc = train(model, optimizer, criterion, trainloader)
        print("Train Time: %.3f" % (time.time() - tic))
        if train_acc > best_train_acc:
            best_train_acc = train_acc

        # Evaluate
        tic = time.time()
        test_acc = test(model, optimizer, criterion, trainloader)
        print("Test Time: %.3f" % (time.time() - tic))
        if test_acc > best_test_acc:
            best_test_acc = test_acc

    print("Best Training Accuracy: %.3f%%" % (best_train_acc * 100.))
    print("Best Test Accuracy: %.3f%%" % (best_test_acc * 100.))

    # Save model and optimizer as checkpoint
    save_dir = getcwd() + "/saved_model.pth"
    save_data = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(save_data, save_dir)
