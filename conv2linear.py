  def conv2linear(IMG_SIZE,kernel_size,padding,stride,num_layers=3,num_nodes=[]):
        for layer in range(num_layers):
            if layer == 0:
                layershape=np.hstack((int(((IMG_SIZE[0]-kernel_size+2*padding)/stride)+1),(int(((IMG_SIZE[1]-kernel_size+2*padding)/stride)+1))))
                poolshape=np.hstack(((layershape/2),num_nodes[layer])).astype(int)
            else:
                layershape=np.hstack((int(((poolshape[0]-kernel_size+2*padding)/stride)+1),(int(((poolshape[1]-kernel_size+2*padding)/stride)+1))))
                poolshape=np.hstack(((layershape/2),num_nodes[layer])).astype(int)
        return np.prod(poolshape)
