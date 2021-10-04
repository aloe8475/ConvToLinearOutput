# ConvToLinearOutput
A simple function to flatten and convolutional layer outputs into a 1D input for linear layers 

**Inputs** - Example values taken from https://pythonprogramming.net/convnet-model-deep-learning-neural-network-pytorch/?completed=/convolutional-neural-networks-deep-learning-neural-network-pytorch/ 
- Initial image size, W = 50, L = 50
- Kernel Size, k = 5
- Stride , s = 1
- Padding, P = 0
- Num Layers = num_layers
- Num Nodes per Layer, q = conv layer output size = num_nodes (list of all the node layer sizes) e.g [32, 64, 128]


To flatten the convolutional layer outputs into inputs for the linear layers, we use the following algorithm:
       

<img src="https://render.githubusercontent.com/render/math?math=O = ({ \frac{(W - k + 2P)}{s} } + 1) *({ \frac{(L - k + 2P)}{s} } + 1)* q">
so, number of pixels/features after 1st layer (q= 32): (((50 - 5 + 0)/1) +1) x (((50 - 5 + 0)/1) +1) x 32 = 46x46x32

==> then we have pooling of {2,2}, so 24/2 = 23x23x32

==> then another layer sized 64: { (23 - 5 + 0)/1 } +1 = 19x19x64

==> then we have pooling of {2,2}, so 24/2 = 9x9x64

==> then another layer sized 128: { (9 - 5 + 0)/1 } +1 = 5x5x128

==> then we have pooling of {2,2}, so 24/2 = 2x2x128 = 512

