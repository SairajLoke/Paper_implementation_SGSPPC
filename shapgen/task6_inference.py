

import torch
import torch.optim as optim
import open3d as o3d
import numpy as np  

from task4_GAN_model import Generator, Discriminator
from utils import draw_point_cloud, save_column_matrix_as_pcd

from configs import OPTIMIZED_SORTED_U_FILEPATH, GENERATOR_MODEL_DIR
from configs import OPTIMIZED_SIGMA_FILEPATH, PLOTPATH_GENERATED, BASIS_SIZE

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_shape():
    # Load the model
    
    generator_id= 1
    gen_epochs = 8
    GENERATOR_MODEL_PATH = f'{GENERATOR_MODEL_DIR}/netG_tid{generator_id}_e{gen_epochs}.pth' # training id 0, epoch 9
    #0 represents the architectural changes if any

    generator = Generator(noise_size=BASIS_SIZE, vt_size=BASIS_SIZE).to(device)
    generator.load_state_dict(torch.load(GENERATOR_MODEL_PATH), strict=True)
    generator.eval()
    
     #--------------------------------------------------------
    # seems good, checked the viz, uncomment to see the viz
    # make sure to have the following configs
   

        #torch.as_tensor() can be used to convert numpy arrays to tensors without copying but still...
    U = torch.tensor( np.load(OPTIMIZED_SORTED_U_FILEPATH), dtype=torch.float32).to(device) 
    print('U', U.shape)
    Sig = torch.tensor(np.load(OPTIMIZED_SIGMA_FILEPATH), dtype=torch.float32).to(device)
    print('Sig', Sig.shape)
    # --------------------------------------------------------

    pidx = 0

    # Generate a shape
    with torch.no_grad():
        #we want (-1,1)
        noise = torch.tensor(np.random.default_rng().uniform(-1,1,(1,BASIS_SIZE)), dtype=torch.float32).to(device)
        vtcol = generator(noise).cpu().numpy()
        #shape is coefficents of basis functions
        print('vtvol shape',vtcol.shape) #be 100x1
        
        p = U@Sig@vtcol.T
        p = p.cpu().detach().numpy()
        print('single generated point cloud p', type(p), p.shape) #be 3000x1
        draw_point_cloud(p)
        point_cloud_save_path = f'{PLOTPATH_GENERATED}/gen_ptcld_g{generator_id}_e{gen_epochs}_pidx{pidx}.png'
        save_column_matrix_as_pcd(point_cloud_save_path, p)




if __name__ == '__main__':
        
    # Load the data
    #optimized_sorted_point_cloud is the data (like 1 image = 1 pt cloud)
    generate_shape()



