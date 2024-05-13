

import torch
import torch.optim as optim
import open3d as o3d
import numpy as np  

from task4_GAN_model import Generator, Discriminator
from utils import draw_point_cloud

from configs import OPTIMIZED_SORTED_U_FILEPATH, GENERATOR_MODEL_PATH


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_shape():
    # Load the model
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(GENERATOR_MODEL_PATH))
    generator.eval()
    
    optimized_U = None

    with open(OPTIMIZED_SORTED_U_FILEPATH, 'rb') as f:
        optimized_U = np.load(f)
        print('optimized_U',optimized_U.shape) #be 3000x100

    # Generate a shape
    with torch.no_grad():
        noise = torch.rand(1, 100).to(device)
        Vt = generator(noise).cpu().numpy()
        #shape is coefficents of basis functions
        print('shape',Vt.shape) #be 100x1
        S = torch.ones(100, 100).to(device) #should be loaded as well

        shape = U@S@Vt
        draw_point_cloud(shape)

if __name__ == '__main__':
        
    # Load the data
    #optimized_sorted_point_cloud is the data (like 1 image = 1 pt cloud)
    with open(OPTIMIZED_SORTED_U_FILEPATH, 'rb') as f:
        optimized_U = np.load(f)
        print('optimized_U',optimized_U.shape)#, optimized_U, '\n-------------------')
        print(type(optimized_U))

        #viz some pt clouds and color the pts, to confirm the pts loaded correctly
        draw_point_cloud(optimized_U) #works



