# ConvToLinearOutput
A simple function to flatten and convolutional layer outputs into a 1D input for linear layers in PyTorch

## Required Packages:
- Python
- PyTorch
- Numpy

## Inputs

Example values taken from https://pythonprogramming.net/convnet-model-deep-learning-neural-network-pytorch/?completed=/convolutional-neural-networks-deep-learning-neural-network-pytorch/ 

- Initial image size, (W = 50, L = 50) ==> IMG_SIZE (list of width x length) e.g. [50,50]
- Kernel Size, k = 5  ==> kernel_size
- Stride , s = 1 ==> stride
- Padding, P = 0 ==> padding
- Num Layers = 3 ==> num_layers
- Num Nodes per Layer, q = (32,64,128) ==> num_nodes (list of number of nodes in each convolutional layer) e.g [32, 64, 128]


To flatten the convolutional layer outputs into inputs for the linear layers, we use the following algorithm:
       
## Algorithm

<img src="https://render.githubusercontent.com/render/math?math=O = ({ \frac{(W - k + 2P)}{s} } + 1) *({ \frac{(L - k + 2P)}{s} } + 1)* q">

So, using the example values, the number of pixels/features after 1st layer (q= 32): (((50 - 5 + 0)/1) +1) x (((50 - 5 + 0)/1) +1) x 32 = 46x46x32

==> then we have pooling of {2,2}, so 24/2 = 23x23x32

==> then another layer sized 64: { (23 - 5 + 0)/1 } +1 = 19x19x64

==> then we have pooling of {2,2}, so 24/2 = 9x9x64

==> then another layer sized 128: { (9 - 5 + 0)/1 } +1 = 5x5x128

==> then we have pooling of {2,2}, so 24/2 = 2x2x128 = 512

The output of conv2linear(<inputs>) will be 512, which you can then use for your linear layer.
       
## Example Implementation
See [example](example.py) 
