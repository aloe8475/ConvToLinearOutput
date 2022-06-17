  def conv2linear(IMG_SIZE,kernel_size,padding,stride,pool_size,pool_padding,pool_stride,num_layers=3,num_nodes=[]):
        for layer in range(num_layers):
            if layer == 0: #if this is the first layer, we want the image size
                layershape=np.hstack((int(((IMG_SIZE[0]-kernel_size+2*padding)/stride)+1),(int(((IMG_SIZE[1]-kernel_size+2*padding)/stride)+1))))
                poolshape=np.hstack((int(((layershape[0]-pool_size+2*pool_padding)/pool_stride)+1),(int(((layershape[1]-pool_size+2*pool_padding)/pool_stride)+1)))).astype(int)
            else: #if this is layer 2+, we want the previous output pool layer size 
                layershape=np.hstack((int(((poolshape[0]-kernel_size+2*padding)/stride)+1),(int(((poolshape[1]-kernel_size+2*padding)/stride)+1))))
                poolshape=np.hstack((int(((layershape[0]-pool_size+2*pool_padding)/pool_stride)+1),(int(((layershape[1]-pool_size+2*pool_padding)/pool_stride)+1)))).astype(int)
        return np.prod(poolshape) 
