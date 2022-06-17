# ConvToLinearOutput
A simple function to calculate 1D input for linear layers from convolutional layer outputs in PyTorch

## Required Packages:
- Python
- PyTorch
- Numpy

## Inputs

Example values taken from https://pythonprogramming.net/convnet-model-deep-learning-neural-network-pytorch/?completed=/convolutional-neural-networks-deep-learning-neural-network-pytorch/ 

- Initial image size, (WxL) ==> IMG_SIZE (list of width x length), e.g. [50,50]
- Kernel Size, k ==> kernel_size, e.g. 5
- Stride , s ==> stride, e.g. 1
- Padding, P ==> padding, e.g. 0
- Num Layers ==> num_layers, e.g. 3
- Num Nodes per Layer, q = (32,64,128) ==> num_nodes (list of number of nodes in each convolutional layer), e.g [32, 64, 128]


To flatten the convolutional layer outputs into inputs for the linear layers, we use the following algorithm:
       
## Algorithm

<img src="https://render.githubusercontent.com/render/math?math=O = ({ \frac{(W - k \%2b 2P)}{s} } + 1) *({ \frac{(L - k \%2b 2P)}{s} } + 1)* q">

So, using the example values, the number of pixels/features after 1st layer (q= 32): (((50 - 5 + 0)/1) +1) x (((50 - 5 + 0)/1) +1) x 32 = 46x46x32

==> then we have pooling of {2,2}, so (((46 - 2 + 0)/2) +1) x (((46 - 2 + 0)/2) +1) x 32 = 23x23x32

==> then another layer sized 64: { (23 - 5 + 0)/1 } +1 = 19x19x64

==> then we have pooling of {2,2}: { (19 - 2 + 0)/2 } +1==> 9x9x64

==> then another layer sized 128: { (9 - 5 + 0)/1 } +1 = 5x5x128

==> then we have pooling of {2,2}, so { (5 - 2 + 0)/2 } +1 = 2x2x128 = 512 (flattened)

The output of conv2linear(<inputs>) will be 512, which you can then use for your linear layer.
       
## Example Implementation
See [example](example.py) - conv2linear() is on line 26
