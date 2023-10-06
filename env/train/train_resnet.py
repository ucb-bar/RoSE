import torch.nn as nn
import torch
from torch import Tensor
from typing import Type

import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt

import torch.onnx
import onnx

import os
from os import walk
import sys

from torchvision import datasets, transforms
import cv2

from resnet import ResNet, BasicBlock, Bottleneck, MULTIHEAD

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import torch.nn.functional as nnf

BATCH = 32
INPUT_DIM = 56
INPUT_DIMX = 320
INPUT_DIMY = 180

def scale_images(img):
    return  nnf.interpolate(img, size=(INPUT_DIMX, INPUT_DIMY), mode='bicubic', align_corners=False)

def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    running_correct = 0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    if MULTIHEAD:
        training_loader = zip(training_loader_angular, training_loader_lateral)
    else:
        training_loader = zip(training_loader)
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        loss = 0 
        optimizer.zero_grad()
        for head, dat in enumerate(data):
            inputs, labels = dat
            # inputs, labels = [dat.to(device) for dat in data]
            inputs = scale_images(inputs)
            # print(f"inputs: {inputs.shape}")
            inputs = inputs.to(device)

            # print(data)

            for j in range(len(labels)):
                if labels[j] == 1:
                    labels[j] = 2
                elif labels[j] == 2:
                    labels[j] = 1
            # print(f"after: {labels}")
            labels = labels.to(device)

            # Zero your gradients for every batch!

            # Make predictions for this batch
            outputs = model(inputs)[:,3*head:3*head+3]
            output_labels = torch.argmax(outputs, dim=1)
            batch_correct = torch.sum(output_labels == labels)
            running_correct += batch_correct / (BATCH * len(data))
            loss = loss + loss_fn(outputs, labels).to(device)
        #loss = 0 
        #for i in range(len(labels)):
        #    loss += loss_fn(outputs[i], truth[i]).to(device)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i%10 == 9:
            print(f"  Running batch {i+1}, running_loss = {running_loss / ((i % 100)+1)}, frac_correct = {running_correct / ((i % 100)+1)}")
        if i % 100 == 99:
            last_loss = running_loss / 100 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(list(training_loader)) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
            running_correct = 0

    return last_loss

if __name__ == "__main__":

    transform = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])

    train_dataset_angular = datasets.ImageFolder('./env/dataset/multihead_data/train/angular/', transform=transform)
    test_dataset_angular = datasets.ImageFolder('./env/dataset/multihead_data/test/angular/', transform=transform)
    training_loader_angular = torch.utils.data.DataLoader(train_dataset_angular, batch_size=BATCH, shuffle=True)
    testing_loader_angular = torch.utils.data.DataLoader(test_dataset_angular, batch_size=BATCH, shuffle=True)

    train_dataset_lateral = datasets.ImageFolder('./env/dataset/multihead_data/train/lateral/', transform=transform)
    test_dataset_lateral = datasets.ImageFolder('./env/dataset/multihead_data/test/lateral/', transform=transform)
    training_loader_lateral = torch.utils.data.DataLoader(train_dataset_lateral, batch_size=BATCH, shuffle=True)
    testing_loader_lateral = torch.utils.data.DataLoader(test_dataset_lateral, batch_size=BATCH, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dnn_name = "resnet14"
    if len(sys.argv) > 1:
        if "6" in sys.argv[1]:
            dnn_name = "resnet6"
            model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=3, num_layers=2).to(device)
        elif "11" in sys.argv[1]:
            dnn_name = "resnet11"
            model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=3, num_layers=3).to(device)
        elif "14" in sys.argv[1]:
            dnn_name = "resnet14"
            model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=3, num_layers=4).to(device)
        elif "18" in sys.argv[1]:
            dnn_name = "resnet18"
            model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=3).to(device)
        elif "34" in sys.argv[1]:
            dnn_name = "resnet34"
            model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=3).to(device)
        else:
            dnn_name = "resnet50"
            model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=3).to(device)
    else:
        dnn_name = "resnet14"
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=3).to(device)

    # model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=3).to(device)
    # model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=3).to(device)
    # model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=3).to(device)
    # model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=3).to(device)
    # model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=3, num_layers=4).to(device)
    # model = AlexNetDualHead(num_classes=3).to(device)

    x = torch.randn(1, 3, INPUT_DIM, INPUT_DIM, requires_grad=True).to(device)
    input = np.random.random((1,3,INPUT_DIM,INPUT_DIM)).astype(np.float32)

    torch_out = model(x)

    # loss_fn = torch.nn.CrossEntropyLoss().to(device)
    loss_fn = torch.nn.NLLLoss().to(device)

    # Optimizers specified in the torch.optim package
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 4

    best_vloss = 1_000_000.
    print(f"Memory usage pre-training: {torch.cuda.memory_allocated(0)}")
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)

        # We don't need gradients on to do reporting
        model.train(False)

        print(f"Memory usage post-training: {torch.cuda.memory_allocated(0)}")
        torch.cuda.empty_cache()
        print(f"Memory usage pre-validation: {torch.cuda.memory_allocated(0)}")
        running_vloss = 0.0
        running_correct = 0
        i = 1
        if MULTIHEAD:
            testing_loader = zip(testing_loader_angular, testing_loader_lateral)
        else:
            testing_loader = zip(testing_loader)
        for i, vdata in enumerate(testing_loader):
            for head, vdat in enumerate(vdata):
                vinputs, vlabels = vdat
                for j in range(len(vlabels)):
                    if vlabels[j] == 1:
                        vlabels[j] = 2
                    elif vlabels[j] == 2:
                        vlabels[j] = 1
                vinputs = scale_images(vinputs)
                vlabels = vlabels.to(device)
                vinputs = vinputs.to(device)
                voutputs = model(vinputs)[:,3*head:3*head+3]
                vloss = loss_fn(voutputs, vlabels)
                output_labels = torch.argmax(voutputs, dim=1)
                batch_correct = torch.sum(output_labels == vlabels)
                running_correct += batch_correct / (BATCH * len (vdat))
                running_vloss += vloss.item()
        print(f"Memory usage post-validation: {torch.cuda.memory_allocated(0)}")
        torch.cuda.empty_cache()
        print(f"Memory usage post-epoch: {torch.cuda.memory_allocated(0)}")


        avg_vloss = running_vloss / (i + 1)
        # avg_vloss = running_vloss / 1
        print('LOSS train {} Validation {}'.format(avg_loss, avg_vloss))
        print('CORRECT Validation {}'.format(running_correct / (i+1)))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        epoch_number += 1



    outputfile = f"./env/train/trail_dnn_{dnn_name}.onnx"
    torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  outputfile,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=12,          # the ONNX version to export the model to
                  # do_constant_folding=True,  # whether to execute constant folding for optimization
                  do_constant_folding=False,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output name
    )

    ort_sess = ort.InferenceSession(outputfile)
    onnx_model = onnx.load(outputfile)


    outputs = ort_sess.run(None, {'input': input})
    print(f"onnx outputs: {outputs}")
    print(f"torch outputs: {torch_out}")
    # print(onnx_model)

