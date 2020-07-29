import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import pdb
from os import getcwd

from models import CustomNet
from params import Params
from dataloader import prep_dataloaders
from ssl import filter_and_channel_wise_ssl_loss, shape_fiber_ssl_loss


def train_ssl(model, optimizer, ssl_loss_func, trainloader, params):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if params.use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = ssl_loss_func(model, outputs, targets, params)
        loss.backward()
        optimizer.step()

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

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if params.use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    print("Test: Loss: %.3f, Acc: %.3f (%d/%d)" % (test_loss / \
          (batch_idx + 1), correct / total * 100., correct, total))

    return correct / total


def count_sparse_wgt(model, threshold):
    weight_cnt = 0
    sparse_weight_cnt = 0
    with torch.no_grad():
        for param_key in model.state_dict():
            param_tensor = model.state_dict()[param_key]
            dims = 1
            for dim in list(param_tensor.size()):
                dims *= dim
            weight_cnt += dims
            sparse_weight_cnt += torch.sum(param_tensor < threshold).item()
    return weight_cnt, sparse_weight_cnt


def count_sparse_wgt_by_layer(model, threshold):
    wgt_cnts = []
    sparse_wgt_cnts = []
    with torch.no_grad():
        for param_key in model.state_dict():
            param_tensor = model.state_dict()[param_key]
            dims = 1
            for dim in list(param_tensor.size()):
                dims *= dim
            wgt_cnts.append((param_key, dims))
            sparse_wgt_cnt_layer = torch.sum(param_tensor < threshold).item()
            sparse_wgt_cnts.append((param_key, sparse_wgt_cnt_layer))
    return wgt_cnts, sparse_wgt_cnts


def count_sparse_wgt_by_filter(model, threshold):
    sparse_wgt_cnts = []
    with torch.no_grad():
        for param_key in model.state_dict():
            param_tensor = model.state_dict()[param_key]
            if len(param_tensor.size()) != 4:
                sparse_wgt_cnts.append((param_key, None))
                continue
            num_filters = param_tensor.size()[0]
            sparse_wgt_cnts_by_filter = []
            for filter_idx in range(num_filters):
                cnt = torch.sum(param_tensor[filter_idx, :, :, :] < \
                                threshold).item()
                sparse_wgt_cnts_by_filter.append(cnt)
            sparse_wgt_cnts.append((param_key, sparse_wgt_cnts_by_filter))
    return sparse_wgt_cnts


def count_sparse_wgt_by_channel(model, threshold):
    sparse_wgt_cnts = []
    with torch.no_grad():
        for param_key in model.state_dict():
            param_tensor = model.state_dict()[param_key]
            if len(param_tensor.size()) != 4:
                sparse_wgt_cnts.append((param_key, None))
                continue
            num_channels = param_tensor.size()[1]
            sparse_wgt_cnts_by_channel = []
            for channel_idx in range(num_channels):
                cnt = torch.sum(param_tensor[:, channel_idx, :, :] < \
                                threshold).item()
                sparse_wgt_cnts_by_channel.append(cnt)
            sparse_wgt_cnts.append((param_key, sparse_wgt_cnts_by_channel))
    return sparse_wgt_cnts


def print_sparse_weights(model, threshold):
    wgt_cnt, sparse_wgt_cnt = count_sparse_wgt(model, threshold)
    print("\nTotal sparse weights: %.3f (%d/%d)" % (100. * sparse_wgt_cnt / \
          wgt_cnt, sparse_wgt_cnt, wgt_cnt))
    
    wgt_cnts, sparse_wgt_cnts = count_sparse_wgt_by_layer(model, threshold)
    print("\nSparse weight by layer")
    for idx in range(len(wgt_cnts)):
        layer_name = wgt_cnts[idx][0]
        wgt_cnt = wgt_cnts[idx][1]
        sparse_wgt_cnt = sparse_wgt_cnts[idx][1]
        print("Layer: {}, {} ({}/{})".format(layer_name, sparse_wgt_cnt / \
              wgt_cnt, sparse_wgt_cnt, wgt_cnt))
    
    sparse_wgt_cnts = count_sparse_wgt_by_filter(model, threshold)
    print("\nSparse weight by filter")
    for idx in range(len(sparse_wgt_cnts)):
        layer_name = sparse_wgt_cnts[idx][0]
        wgts_filters = sparse_wgt_cnts[idx][1]
        print("Layer: {}, {}".format(layer_name, wgts_filters))


if __name__ == "__main__":
    params = Params()

    trainloader, testloader, classes = prep_dataloaders()
    model = CustomNet(len(classes), use_gpu=params.use_gpu)
    if params.use_gpu:
        model = model.cuda()
    # optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0.9,
    #                             weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    criterion = nn.CrossEntropyLoss()

    checkpoint = torch.load(getcwd() + "/saved_model.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print_sparse_weights(model, params.threshold)

    best_train_acc = 0
    best_test_acc = 0

    for epoch in range(params.num_sparse_train_epochs):
        print("========== epoch %d" % (epoch))
        # if epoch % 50 == 0:
        #     scale_lr(optimizer, 0.1)

        if params.ssl_type == "filter_channel":
            ssl_loss_func = filter_and_channel_wise_ssl_loss
        else:
            ssl_loss_func = shape_fiber_ssl_loss

        tic = time.time()
        train_acc = train_ssl(model, optimizer, ssl_loss_func, trainloader,
                              params)
        print("Train Time: %.3f" % (time.time() - tic))
        if train_acc > best_train_acc:
            best_train_acc = train_acc

        tic = time.time()
        test_acc = test(model, optimizer, criterion, trainloader)
        print("Test Time: %.3f" % (time.time() - tic))
        if test_acc > best_test_acc:
            best_test_acc = test_acc

    print("Best Training Accuracy: %.3f%%" % (best_train_acc * 100.))
    print("Best Test Accuracy: %.3f%%" % (best_test_acc * 100.))

    print_sparse_weights(model, params.threshold)
    test_acc = test(model, optimizer, criterion, trainloader)
    print("Final test accuracy: {}".format(test_acc * 100.))
