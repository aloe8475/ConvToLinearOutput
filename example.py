#This example is taken from https://pythonprogramming.net/convnet-model-deep-learning-neural-network-pytorch/?completed=/convolutional-neural-networks-deep-learning-neural-network-pytorch/

class Net(nn.Module):#input a nn module
    def __init__(self):#initalise empty nn object
        super().__init__()#this inhereits initialization from nn.Module, but also runs a new init method
        #initalize parameters
        kernel_size = 5
        padding = 0
        stride = 1
        IMG_SIZE = [50, 50] #width x length
        num_layers = 3
        num_nodes = [32, 64, 128]
        
        #define convolutional layers
        self.conv1 = nn.Conv2d(1,  32, kernel_size) #1 input, 32 outputs, kernel size =5
        self.conv2 = nn.Conv2d(32, 64, kernel_size) #32 inputs, 64 outputs, kernel size = 5
        self.conv3 = nn.Conv2d(64, 128, kernel_size) #64 inputs, 128 outputs, kernel size = 5
        
        # we now need to move from convolution layers to linear layers (FLATTEN)
        flatConv=self.conv2linear(IMG_SIZE,kernel_size,padding,stride,num_layers,num_nodes=num_nodes) #CALL conv2linear to go from convolutional to linear
        
        #define linear layers
        self.fc1 = nn.Linear(flatConv, 512) #flattening
        self.fc2 = nn.Linear(512, 2)#2 outputs = 2 classes
    
    #conv2linear.py defined in Net class
    def conv2linear(self,IMG_SIZE,kernel_size,padding,stride,num_layers=3,num_nodes=[]):
        for layer in range(num_layers):
            if layer == 0:
                layershape=np.hstack((int(((IMG_SIZE[0]-kernel_size+2*padding)/stride)+1),(int(((IMG_SIZE[1]-kernel_size+2*padding)/stride)+1))))
                poolshape=np.hstack(((layershape/2),num_nodes[layer])).astype(int)
            else:
                layershape=np.hstack((int(((poolshape[0]-kernel_size+2*padding)/stride)+1),(int(((poolshape[1]-kernel_size+2*padding)/stride)+1))))
                poolshape=np.hstack(((layershape/2),num_nodes[layer])).astype(int)
        return np.prod(poolshape)
    
   #define how data passes through Net model (feed-forward)
   def forward(self, x):
          x = self.convs(x)
          x = x.view(-1,self._to_linear) #flatten the data
          x = F.relu(self.fc1(x)) #relu activation for linear layer
          x = self.fc2(x) #output, no activation
          return F.softmax(x, dim=1)

net = Net()
print(net)
