  def conv2linear(IMG_SIZE,kernel_size,padding,stride,num_layers=3,num_nodes=[]):
        for layer in range(num_layers):
            if layer == 0: #if this is the first layer, we want the image size
                layershape=np.hstack((int(((IMG_SIZE[0]-kernel_size+2*padding)/stride)+1),(int(((IMG_SIZE[1]-kernel_size+2*padding)/stride)+1))))
            else: #if this is layer 2+, we want the previous layer size 
                layershape=np.hstack((int(((layershape[0]-kernel_size+2*padding)/stride)+1),(int(((poolshape[1]-kernel_size+2*padding)/stride)+1))))
        return np.prod(layershape) 
