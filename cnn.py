################################################################################
#
# LOGISTICS
#
#    Name: Raisaat Rashid
#    Net ID: rar150430
#
# DESCRIPTION
#
#    Image classification in PyTorch for ImageNet reduced to 100 classes and
#    down sampled such that the short side is 64 pixels and the long side is
#    >= 64 pixels
#
#    This script achieved a best accuracy of 67.32% on epoch 125 with a learning
#    rate at that point of 0.001023 and time required for each epoch of ~ 117 s
#
# NOTES
#
#    0. For a mapping of category names to directory names see:
#       https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57
#
#    1. The original 2012 ImageNet images are down sampled such that their short
#       side is 64 pixels (the other side is >= 64 pixels) and only 100 of the
#       original 1000 classes are kept.
#
#    2. Build and train a RegNetX image classifier modified as follows:
#
#       - Set stride = 1 (instead of stride = 2) in the stem
#       - Replace the first stride = 2 down sampling building block in the
#         original network by a stride = 1 normal building block
#       - The fully connected layer in the decoder outputs 100 classes instead
#         of 1000 classes
#
#       The original RegNetX takes in 3x224x224 input images and generates Nx7x7
#       feature maps before the decoder, this modified RegNetX will take in
#       3x56x56 input images and generate Nx7x7 feature maps before the decoder.
#
################################################################################

################################################################################
#
# IMPORT
#
################################################################################

# torch
import torch
import torch.nn       as     nn
import torch.optim    as     optim
from   torch.autograd import Function

# torch utils
import torchvision
import torchvision.transforms as transforms

# additional libraries
import os
import urllib.request
import zipfile
import time
import math
import numpy             as np
import matplotlib.pyplot as plt

################################################################################
#
# PARAMETERS
#
################################################################################

# data
DATA_DIR_1        = 'data'
DATA_DIR_2        = 'data/imagenet64'
DATA_DIR_TRAIN    = 'data/imagenet64/train'
DATA_DIR_TEST     = 'data/imagenet64/val'
DATA_FILE_TRAIN_1 = 'Train1.zip'
DATA_FILE_TRAIN_2 = 'Train2.zip'
DATA_FILE_TRAIN_3 = 'Train3.zip'
DATA_FILE_TRAIN_4 = 'Train4.zip'
DATA_FILE_TRAIN_5 = 'Train5.zip'
DATA_FILE_TEST_1  = 'Val1.zip'
DATA_URL_TRAIN_1  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train1.zip'
DATA_URL_TRAIN_2  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train2.zip'
DATA_URL_TRAIN_3  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train3.zip'
DATA_URL_TRAIN_4  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train4.zip'
DATA_URL_TRAIN_5  = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Train5.zip'
DATA_URL_TEST_1   = 'https://github.com/arthurredfern/UT-Dallas-CS-6301-CNNs/raw/master/Data/Val1.zip'
DATA_BATCH_SIZE   = 512
DATA_NUM_WORKERS  = 4
DATA_NUM_CHANNELS = 3
DATA_NUM_CLASSES  = 100
DATA_RESIZE       = 64
DATA_CROP         = 56
DATA_MEAN         = (0.485, 0.456, 0.406)
DATA_STD_DEV      = (0.229, 0.224, 0.225)

# model parameters
MODEL_LEVEL_WIDTHS = [24, 56, 152, 368] # Width/output channels of each of the 4 levels in the encoder body
MODEL_LEVEL_DEPTHS = [1, 1, 4, 7] # Depth of/number of blocks in each of the 4 levels in the encoder body
MODEL_BOTTLENECK_RATIO = 1
MODEL_GROUP_SIZE = 8
MODEL_STEM_WIDTH = 24
MODEL_CONV3_FILTER_FR = 3
MODEL_CONV3_FILTER_FC = 3

# training (linear warm up with cosine decay learning rate)
TRAINING_LR_MAX          = 0.001
TRAINING_LR_INIT_SCALE   = 0.01
TRAINING_LR_INIT_EPOCHS  = 5
TRAINING_LR_FINAL_SCALE  = 0.01
TRAINING_LR_FINAL_EPOCHS = 120
TRAINING_NUM_EPOCHS      = TRAINING_LR_INIT_EPOCHS + TRAINING_LR_FINAL_EPOCHS
TRAINING_LR_INIT         = TRAINING_LR_MAX*TRAINING_LR_INIT_SCALE
TRAINING_LR_FINAL        = TRAINING_LR_MAX*TRAINING_LR_FINAL_SCALE

# file parameters
FILE_NAME = 'Model.pt'
FILE_SAVE = 1
FILE_LOAD = 0

################################################################################
#
# DATA
#
################################################################################

# create a local directory structure for data storage
if (os.path.exists(DATA_DIR_1) == False):
    os.mkdir(DATA_DIR_1)
if (os.path.exists(DATA_DIR_2) == False):
    os.mkdir(DATA_DIR_2)
if (os.path.exists(DATA_DIR_TRAIN) == False):
    os.mkdir(DATA_DIR_TRAIN)
if (os.path.exists(DATA_DIR_TEST) == False):
    os.mkdir(DATA_DIR_TEST)

# download data
if (os.path.exists(DATA_FILE_TRAIN_1) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_1, DATA_FILE_TRAIN_1)
if (os.path.exists(DATA_FILE_TRAIN_2) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_2, DATA_FILE_TRAIN_2)
if (os.path.exists(DATA_FILE_TRAIN_3) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_3, DATA_FILE_TRAIN_3)
if (os.path.exists(DATA_FILE_TRAIN_4) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_4, DATA_FILE_TRAIN_4)
if (os.path.exists(DATA_FILE_TRAIN_5) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_5, DATA_FILE_TRAIN_5)
if (os.path.exists(DATA_FILE_TEST_1) == False):
    urllib.request.urlretrieve(DATA_URL_TEST_1, DATA_FILE_TEST_1)

# extract data
with zipfile.ZipFile(DATA_FILE_TRAIN_1, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_2, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_3, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_4, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TRAIN_5, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TRAIN)
with zipfile.ZipFile(DATA_FILE_TEST_1, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR_TEST)

# transforms
transform_train = transforms.Compose([transforms.RandomResizedCrop(DATA_CROP), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize(DATA_MEAN, DATA_STD_DEV)])
transform_test  = transforms.Compose([transforms.Resize(DATA_RESIZE), transforms.CenterCrop(DATA_CROP), transforms.ToTensor(), transforms.Normalize(DATA_MEAN, DATA_STD_DEV)])

# data sets
dataset_train = torchvision.datasets.ImageFolder(DATA_DIR_TRAIN, transform=transform_train)
dataset_test  = torchvision.datasets.ImageFolder(DATA_DIR_TEST,  transform=transform_test)

# data loader
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=DATA_BATCH_SIZE, shuffle=True,  num_workers=DATA_NUM_WORKERS, pin_memory=True, drop_last=True)
dataloader_test  = torch.utils.data.DataLoader(dataset_test,  batch_size=DATA_BATCH_SIZE, shuffle=False, num_workers=DATA_NUM_WORKERS, pin_memory=True, drop_last=True)

################################################################################
#
# NETWORK BUILDING BLOCK
#
################################################################################

# X block
class XBlock(nn.Module):

    # initialization
    def __init__(self, Ni, No, Fr, Fc, Sr, Sc, G):

        # parent initialization
        super(XBlock, self).__init__()
        
        self.downsample = False

        # identity
        if Sr != 1 or Ni != No:
            self.downsample = True
            self.conv0 = nn.Conv2d(Ni, No, kernel_size=1, stride=(Sr, Sc), bias=False)
            self.bn0 = nn.BatchNorm2d(No)
        
        # layers
        self.conv1 = nn.Conv2d(Ni, Ni, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(Ni)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(Ni, Ni, kernel_size=(Fr, Fc), stride=(Sr, Sc), padding=1, groups = Ni // G, bias=False)
        self.bn2 = nn.BatchNorm2d(Ni)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(Ni, No, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(No)
        self.relu3 = nn.ReLU()

    # forward path
    def forward(self, x):
        # residual
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.relu1(res)
        
        res = self.conv2(res)
        res = self.bn2(res)
        res = self.relu2(res)

        res = self.conv3(res)
        res = self.bn3(res)
        
        # identity
        if self.downsample:
            x = self.conv0(x)
            x = self.bn0(x)

        y = res + x
        y = self.relu3(y)

        # return
        return y

################################################################################
#
# NETWORK
#
################################################################################

# define
class Model(nn.Module):

    # initialization
    def __init__(self, data_num_channels, 
                 w, d, g, stem_width, filter_Fr, filter_Fc,
                 data_num_classes):

        # parent initialization
        super(Model, self).__init__()
        
        # encoder stem
        self.stem = nn.ModuleList()
        self.stem.append(nn.Conv2d(data_num_channels, stem_width, kernel_size=3, padding=1, bias=False))
        self.stem.append(nn.BatchNorm2d(stem_width))
        self.stem.append(nn.ReLU())
      
        # encoder body - level 1
        self.enc_1 = nn.ModuleList()
        self.enc_1.append(XBlock(stem_width, w[0], filter_Fr, filter_Fc, 1, 1, g))
        for i in range(d[0] - 1):
          self.enc_1.append(XBlock(w[0], w[0], filter_Fr, filter_Fc, 1, 1, g))

        # encoder body - level 2
        self.enc_2 = nn.ModuleList()
        self.enc_2.append(XBlock(w[0], w[1], filter_Fr, filter_Fc, 2, 2, g))
        for i in range(d[1] - 1):
          self.enc_2.append(XBlock(w[1], w[1], filter_Fr, filter_Fc, 1, 1, g))

        # encoder body - level 3
        self.enc_3 = nn.ModuleList()
        self.enc_3.append(XBlock(w[1], w[2], filter_Fr, filter_Fc, 2, 2, g))
        for i in range(d[2] - 1):
          self.enc_3.append(XBlock(w[2], w[2], filter_Fr, filter_Fc, 1, 1, g))

        # encoder body - level 4
        self.enc_4 = nn.ModuleList()
        self.enc_4.append(XBlock(w[2], w[3], filter_Fr, filter_Fc, 2, 2, g))
        for i in range(d[3] - 1):
          self.enc_4.append(XBlock(w[3], w[3], filter_Fr, filter_Fc, 1, 1, g))
        
        # decoder
        self.dec = nn.ModuleList()
        self.dec.append(nn.AdaptiveAvgPool2d(output_size=1))
        self.dec.append(nn.Flatten())
        self.dec.append(nn.Linear(w[3], data_num_classes))
        
    # forward path
    def forward(self, x):
        
        # forward propagation through the encoder stem
        for layer in self.stem:
            x = layer(x)

        # forward propagation through encoder body - level 1
        for layer in self.enc_1:
            x = layer(x)

        # forward propagation through encoder body - level 2
        for layer in self.enc_2:
            x = layer(x)
        
        # forward propagation through encoder body - level 3
        for layer in self.enc_3:
            x = layer(x)
        
        # forward propagation through encoder body - level 4
        for layer in self.enc_4:
            x = layer(x)
                
        # forward propagation through the decoder
        for layer in self.dec:
            x = layer(x)

        y = x

        # return
        return y

# create model
model = Model(DATA_NUM_CHANNELS, MODEL_LEVEL_WIDTHS, MODEL_LEVEL_DEPTHS, MODEL_GROUP_SIZE, MODEL_STEM_WIDTH, MODEL_CONV3_FILTER_FR, MODEL_CONV3_FILTER_FC, DATA_NUM_CLASSES)

# specify the device as the GPU if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# enable data parallelization for multi GPU systems
if (torch.cuda.device_count() > 1):
   model = nn.DataParallel(model)
print('Using {0:d} GPU(s)'.format(torch.cuda.device_count()), flush=True)

# transfer the network to the device
model.to(device)

################################################################################
#
# ERROR AND OPTIMIZER
#
################################################################################

# error (softmax cross entropy)
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

################################################################################
#
# TRAINING
#
################################################################################

# start epoch
start_epoch = 0

# learning rate schedule
def lr_schedule(epoch):

    # linear warmup followed by cosine decay
    if epoch < TRAINING_LR_INIT_EPOCHS:
        lr = (TRAINING_LR_MAX - TRAINING_LR_INIT)*(float(epoch)/TRAINING_LR_INIT_EPOCHS) + TRAINING_LR_INIT
    else:
        lr = (TRAINING_LR_MAX - TRAINING_LR_FINAL)*max(0.0, math.cos(((float(epoch) - TRAINING_LR_INIT_EPOCHS)/(TRAINING_LR_FINAL_EPOCHS - 1.0))*(math.pi/2.0))) + TRAINING_LR_FINAL

    return lr

# model loading
if FILE_LOAD == 1:
    checkpoint = torch.load(FILE_NAME)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

# cycle through the epochs
epochs = []
accuracies = []
losses = []
start_time = time.time()
for epoch in range(start_epoch, TRAINING_NUM_EPOCHS):
    epochs.append(epoch)

    # initialize train set statistics
    model.train()
    training_loss = 0.0
    num_batches   = 0

    # set the learning rate for the epoch
    for g in optimizer.param_groups:
        g['lr'] = lr_schedule(epoch)

    # cycle through the train set
    for data in dataloader_train:

        # extract a batch of data and move it to the appropriate device
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass, loss, backward pass and weight update
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # update statistics
        training_loss = training_loss + loss.item()
        num_batches   = num_batches + 1

    # initialize test set statistics
    model.eval()
    test_correct = 0
    test_total   = 0

    # no weight update / no gradient needed
    with torch.no_grad():

        # cycle through the test set
        for data in dataloader_test:

            # extract a batch of data and move it to the appropriate device
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # forward pass and prediction
            outputs      = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # update test set statistics
            test_total   = test_total + labels.size(0)
            test_correct = test_correct + (predicted == labels).sum().item()

    # epoch statistics
    accuracy = 100.0*test_correct/test_total
    accuracies.append(accuracy)
    loss = (training_loss/num_batches)/DATA_BATCH_SIZE
    losses.append(loss)
    print('Epoch {0:2d} lr = {1:8.6f} avg loss = {2:8.6f} accuracy = {3:5.2f}'.format(epoch, lr_schedule(epoch), loss, accuracy))
    
    # model saving
    if FILE_SAVE == 1:
      torch.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict()
      }, FILE_NAME)

print("\nTotal training time: %s minutes" % ((time.time() - start_time)/60))

################################################################################
#
# TEST
#
################################################################################

# initialize test set statistics
model.eval()
test_correct = 0
test_total   = 0

# no weight update / no gradient needed
with torch.no_grad():

    # cycle through the test set
    for data in dataloader_test:

        # extract a batch of data and move it to the appropriate device
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # forward pass and prediction
        outputs      = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        # update test set statistics
        test_total   = test_total + labels.size(0)
        test_correct = test_correct + (predicted == labels).sum().item()

# test set statistics
print('Final accuracy of test set = {0:5.2f}'.format((100.0*test_correct/test_total)))
print('')

################################################################################
#
# DISPLAY
#
################################################################################

# plot training data loss vs. epoch
plot1 = plt.figure(1)
plt.plot(epochs, losses)
plt.title('Training data loss vs. epoch')
plt.xlabel("Epoch")
plt.ylabel("Training data loss")

# plot training data accuracy vs. epoch
plot1 = plt.figure(2)
plt.plot(epochs, accuracies)
plt.title('Training data accuracy vs. epoch')
plt.xlabel("Epoch")
plt.ylabel("Training data accuracy (%)")

plt.show()
