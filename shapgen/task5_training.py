
import torch
import torch.optim as optim
import open3d as o3d
import numpy as np  

from task4_GAN_model import Generator, Discriminator
from task4_Dataset import ShapeDataset

from configs import TRAINING_DATA_PATH, NUM_EPOCHS, LR, REAL_LABEL, FAKE_LABEL,DISC_K_ITERS, BASIS_SIZE, BATCH_SIZE, BETA1
from utils import draw_point_cloud

from tqdm import tqdm


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the data
    #optimized_sorted_point_cloud is the data (like 1 image = 1 pt cloud)
    #load the data

    pcddataset = ShapeDataset(training_data_path=TRAINING_DATA_PATH)
    pcdloader = torch.utils.data.DataLoader(pcddataset, batch_size=BATCH_SIZE, shuffle=True)

    # test_data
    # real_batch = next(iter(pcdloader)).to(device)
    # for p in real_batch:
    #     pcd = o3d.geometry.PointCloud()
    #     
    #     p = p.reshape(-1,3)
    #     print(p.shape) 
    #     p = p.cpu().detach().numpy()
    #     print(type(p))
    #     pcd.points = o3d.utility.Vector3dVector(p)
    #     draw_point_cloud(pcd)
    
    


    #build the models
    netG = Generator(noise_size=BASIS_SIZE,pcd_size=BASIS_SIZE).to(device)
    print(netG)
    #model init

    netD = Discriminator(pcd_size=BASIS_SIZE).to(device)
    print(netD)
    
    netG.train()
    netD.train()

    optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))

    Gene_losses = []
    Disc_losses = []
    for epoch in range(NUM_EPOCHS):

        for i, data in enumerate(pcdloader,0):

            if data is None:
                print(f'data is none {epoch} {i}')
                continue
            # Training the Discriminator for K_disc iterations
            #----------------------------------------------------------------   
            for k in range(DISC_K_ITERS):
                print('training disc')
                netD.zero_grad()
                print(type(data[0][0]))
                #real data
                # labelr = REAL_LABEL.to(device)
                output_real = netD(data)
                print('real Disc ouput',output_real.shape, output_real)
                if torch.isnan(output_real).any():
                    print('output real contains nan')
                    break

                # errD_real = criterion(output, REAL_LABEL)
                lossD_real_v = torch.log(output_real)
                # D_x = output.mean()
                lossD_real = lossD_real_v.mean()
                # lossD_real.backward()
                #mean before or after log
                # 

                
                #fake data--------------------------------------
                #check if using numpy random is okay or can we gen directly in torch.float32
                noise = torch.tensor(np.random.default_rng().uniform(-1,1,(BATCH_SIZE,BASIS_SIZE)), dtype=torch.float32).to(device)
                fake_pcd = netG(noise)
                 #check the values also 

                output_fake = netD(fake_pcd.detach()) #detach to avoid training generator here
                print('fake disc output', fake_pcd.shape, output_fake)
                lossD_fake_v = torch.log(1 - output_fake) #check 
                lossD_fake = lossD_fake_v.mean()
                # lossD_fake.backward()
                # D_G_z1 = output.mean().item()
                lossD = lossD_real + lossD_fake
                lossD = -1*lossD #mi

                lossD.backward()
                optimizerD.step()

                # running_loss += lossD.item() #TODO

            #----------------------------------------------------------------
            # Training the Generator
            print('training gen')
            # netG.zero_grad()
            # labelf = FAKE_LABEL.to(device)
            # # z = torch.randn(label.size(0), nz, 1, 1, device=device)
            # z = torch.tensor(urng(-1,1,(data.shape))).to(device)
            # output = netG(z)
            # lossG = 

        # Gene_losses.append(lossG)
        Disc_losses.append(lossD)
        
        print('$$$--------------------epoch idx:',epoch,  'lossD:',lossD, '--------------------$$$') #'lossG:',lossG)


    

