# Modified-RegNetX-200MF
This project uses PyTorch to modify, train and implement RegNetX-200MF for image classification of a modified variant of ImageNet. The dataset was created by down-sampling the original ImageNet images such that their short side is 64 pixels (while the other side is >= 64 pixels) and only 100 of the original 1000 classes were kept.

**NOTE:** This project was focused on understanding the different building block structures of a CNN for image classification, not to improve classification accuracy.

## Modified Architecture

* Set stride = 1 (instead of stride = 2) in the stem
* Replace the first stride = 2 down-sampling building block in the original network by a stride = 1 normal building block
* The fully connected layer in the decoder outputs 100 classes instead of 1000 classes
* All of the other blocks in RegNetX-200MF stay the same

The original RegNetX-200MF takes in 3x224x224 input images and generates Nx7x7 feature maps before the decoder. This modified RegNetX-200MF will take in 3x56x56 input images (cropped from the provided data set) and generate Nx7x7 feature maps before the decoder.

## Training

Number of epochs = 125  
Batch size = 512  
Learning rate = linear warmup for 5 epochs followed by cosine decay for 120 epochs  
Error criterion = softmax cross entropy  
Optimizer = Adam  
Weight decay (l2 norm) = 0

Final accuracy = 67.32%

_Refer to cnn.pdf for more details_

## Instructions on running the code

1. Go to Google Colaboratory: https://colab.research.google.com/notebooks/welcome.ipynb
2. File - New Python 3 notebook
3. Cut and paste this file into the cell (feel free to divide into multiple cells)
4. Runtime - Run all
