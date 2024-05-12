

'''
matrix V from PCA provides a compact and yet accurate approximation of the 3D shapes. 
generative model iss to learn to generate the shape coefficients.

model learns V (assuming its dim is 100x5000) basis = 100
V = 100x1 for each shape

so gan model will generate V ( the shape coefficients) for each shape
these coeff when multiplied witht the U (shape basis 3000x100)
                                                 basically each basis is like a basic shape 
                                                to build up the final shape

so U is fixed

#TODO:

1. check weight initialization
2. check batchnorm
3. check loss function


#avoiding sequential model as it is not flexible enough for 2nd loss fn

#maybe we could have decreased z?

#what about last layer in discriminator? shouldnt it be 1?

#training checklist
1. Construct different mini-batches for real and fake, 
    i.e. each mini-batch needs to contain only all real images or all generated images

2. Label smoothening for discriminator (0.9, 0.1)
'''
import torch
import torch.nn as nn

class Generator(nn.Module):

    def __init__(self, noise_size, pcd_size):
        super(Generator, self).__init__()
        self.pcd_size = pcd_size
        self.noise_size = noise_size    
        self.fc1 = nn.Linear(self.noise_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, self.pcd_size)

        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(100)
        self.bn3 = nn.BatchNorm1d(100)
        self.bn4 = nn.BatchNorm1d(self.pcd_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        print('Generator input x', x.shape)

        out = self.fc1(x) #size of last dim of input should match the input size for a linear layer
        print('fc1', self.fc1)
        print('fc1+', out.shape)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.fc4(out)
        out = self.bn4(out)
        out = self.relu(out)

        print('Generator output', out.shape, out)
        return out

class Discriminator(nn.Module):

    def __init__(self, pcd_size):
        super(Discriminator, self).__init__()
        self.pcd_size = pcd_size
        self.fc1 = nn.Linear(self.pcd_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 1)

        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(100)
        self.bn3 = nn.BatchNorm1d(100)
        self.bn4 = nn.BatchNorm1d(100)

        self.leakyrelu = nn.LeakyReLU(0.1)
        self.sigmod = nn.Sigmoid()


    def forward(self, x):

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.leakyrelu(out)
        activation1 = out

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.leakyrelu(out)
        activation2 = out

        out = self.fc3(out)
        out = self.bn3(out)
        out = self.leakyrelu(out)
        activation3 = out

        out = self.fc4(out)
        out = self.bn4(out)
        out = self.leakyrelu(out)
        activation4 = out
        print(activation4.shape)
        #shouldnt the output be 1?
        out = self.fc5(out)
        out = self.sigmod(out)

        activation1 = torch.unsqueeze(activation1,2)
        activation2 = torch.unsqueeze(activation2,2)
        activation3 = torch.unsqueeze(activation3,2)
        activation4 = torch.unsqueeze(activation4,2)
        activations = torch.cat((activation1,activation2,activation3,activation4),dim=-1)

        print('(batchsize x BASIS_SIZE x 4) Activations', activations.shape)

        return out,activations