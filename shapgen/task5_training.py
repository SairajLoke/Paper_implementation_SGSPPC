
import torch
import torch.optim as optim
import open3d as o3d
import numpy as np  

from task4_GAN_model import Generator, Discriminator
from task4_Dataset import ShapeDataset

from configs import TRAINING_DATA_PATH, NUM_EPOCHS, LRD,LRG, REAL_LABEL, FAKE_LABEL,DISC_K_ITERS,  BATCH_SIZE, BETA1
from utils import draw_point_cloud, plot_losses

from configs import GENERATOR_MODEL_DIR, DISCRIMINATOR_MODEL_DIR
from configs import NUM_PT_FEATURES, WIDTH,BASIS_SIZE
from configs import OPTIMIZED_SORTED_U_FILEPATH, OPTIMIZED_SIGMA_FILEPATH
from tqdm import tqdm



device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_generator_loss(activations_fake, activations_real):
    #check if they are on same device 
    
    print('generator loss ', 'device', device)
    exp_activations_fake = activations_fake.mean(dim=0) #check dim
    exp_activations_real = activations_real.mean(dim=0)
    mod1 = torch.norm(exp_activations_fake - exp_activations_real, p=2).to(device)

    torch_covs_fake = torch.zeros(BATCH_SIZE,BASIS_SIZE,BASIS_SIZE).to(device)
    torch_covs_real = torch.zeros(BATCH_SIZE,BASIS_SIZE,BASIS_SIZE).to(device)

    #TODO check if cov dim is correct (seems like, cov is among dims and not across exmaples)
    print(activations_fake.shape, activations_real.shape)
    for i in range(BATCH_SIZE):
        val = torch.cov(activations_fake[i])
        print('f cov shape', val.shape)
        torch_covs_fake[i] = val

    for i in range(BATCH_SIZE):
        val = torch.cov(activations_real[i])
        print('r cov shape', val.shape)
        torch_covs_real[i] = val

    mod2 = torch.tensor(0,dtype=torch.float64 ).to(device)
    for i in range(BATCH_SIZE):
        mod2 += torch.norm(torch_covs_fake[i] - torch_covs_real[i], p=2)
    # cov_fake, cov_real = torch.cov(activations_fake), torch.cov(activations_real) #rowvar not here 
    # mod1 = torch.mean(torch, dim=0)
    print('mod1.shape: ',mod1.shape, 'mod2.shape', mod2.shape)
    assert mod2.get_device() == activations_fake.get_device(), print('mod2 device:',mod2.get_device(), 'device:',device)
    assert mod1.get_device() == activations_fake.get_device(), print('mod1 device:',mod1.get_device(), 'device:',device)
    return mod1 + mod2



if __name__ == '__main__':
    
    # Load the data
    #optimized_sorted_point_cloud is the data (like 1 image = 1 pt cloud)
    #load the data

    #check if the shape coeff dimensions are what? 100 or dimensionx100
    vtdataset = ShapeDataset(training_data_path=TRAINING_DATA_PATH)
     #the Vt matrix 100x2000 (2000 = Shapes, 100 = basis) 2k as 5k was taking too long in svd
    print(f'Loaded vt dataset of len {vtdataset.__len__()} ')  

    vtloader = torch.utils.data.DataLoader(vtdataset, batch_size=BATCH_SIZE, shuffle=True)

    # test the vt data by visualizing it using the U and Sigma matrices stored in the configs
    real_batch = next(iter(pcdloader)).to(device)

    for vtcol in real_batch:
        print('vtcol', vtcol.shape)

        U = np.load(OPTIMIZED_SORTED_U_FILEPATH)
        print('U', U.shape)
        Sig = np.load(OPTIMIZED_SIGMA_FILEPATH)
        print('Sig', Sig.shape)

        p = U@Sig@vtcol

        p = p.reshape(-1,3)
        print(p.shape) 
        p = p.cpu().detach().numpy()
        print(type(p))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p)
        draw_point_cloud(pcd)
    
    


    # #build the models
    # netG = Generator(noise_size=BASIS_SIZE,pcd_size=BASIS_SIZE).to(device)
    # print(netG)
    # #model init

    # netD = Discriminator(pcd_size=NUM_PT_FEATURES*WIDTH ).to(device)
    # print(netD)
    
    # netG.train()
    # netD.train()

    # optimizerD = optim.Adam(netD.parameters(), lr=LRD, betas=(BETA1, 0.999))
    # optimizerG = optim.Adam(netG.parameters(), lr=LRG, betas=(BETA1, 0.999))

    # Gene_losses = []
    # Disc_losses = []


    # for epoch in range(NUM_EPOCHS):
    #     epoch_lossD = torch.tensor(0.0,dtype=torch.float32).to(device)
    #     epoch_lossG = torch.tensor(0.0,dtype=torch.float32).to(device)
    #     for iteridx , data in enumerate(pcdloader,0):
            
    #         # trainD = False

    #         if data is None:
    #             print(f'data is none {epoch} {i} $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    #             continue
    #         # Training the Discriminator for K_disc iterations
    #         #----------------------------------------------------------------  
    #         # #put a accuracy check on discriminator (train if accuracy < 80)
    #         # running_loss = 0.0 
    #         # for k in range(DISC_K_ITERS):

    #         print('training disc')
    #         netD.zero_grad()
    #         print(type(data[0][0]))
    #         #real data
    #         # labelr = REAL_LABEL.to(device)
    #         output_real, activns_real = netD(data)
    #         print('real Disc ouput',output_real.shape, output_real)

    #         #check accuracy of discriminator-----------------------------
    #         # test_output_fake, test_activns_fake = netD(data)
    #         # print('test output fake', test_output_fake.shape, test_output_fake)
    #         # test_output_real = torch.where( output_real > 0.5, torch.tensor(1), torch.tensor(0))
    #         # test_output_fake = torch.where( test_output_fake > 0.5, torch.tensor(1), torch.tensor(0))
    #         # accuracy =  (torch.count_nonzero() + test_output_fake.size - torch.count_nonzero())/(test_output_fake.size + test_output_real.size)
    #         # ones = (test_output_real == 1.).sum(dim=0)#check dim 
    #         # zeros = (test_output_fake == 0.).sum(dim=0)
    #         # accuracy = (ones + zeros)/(test_output_fake.shape(0) + test_output_real.shape(0))
    #         # if(accuracy < 0.8):
    #         #     trainD = True
    #         # -----------------------------------------------------------

    #         if torch.isnan(output_real).any():
    #             print('output real contains nan')
    #             break

    #         # errD_real = criterion(output, REAL_LABEL)
    #         lossD_real_v = torch.log(output_real)
    #         # D_x = output.mean()
    #         lossD_real = lossD_real_v.mean() #expectation(avg) of Log(D(x)) over the batches
    #         # lossD_real.backward()
    #         #mean before or after log
    #         # 

            
    #         #fake data--------------------------------------
    #         #check if using numpy random is okay or can we gen directly in torch.float32
    #         noise = torch.tensor(np.random.default_rng().uniform(-1,1,(BATCH_SIZE,BASIS_SIZE)), dtype=torch.float32).to(device)
    #         fake_pcd = netG(noise)
    #             #check the values also 

    #         output_fake, activns_fake_ignore = netD(fake_pcd.detach()) 
    #         #detach to avoid training generator here,
    #         # So no gradient will be backpropagated to netG along this variable(fake_pcd).
    #         print('fake disc output', output_fake.shape, output_fake)

    #         lossD_fake_v = torch.log(1 - output_fake) #check 
    #         lossD_fake = lossD_fake_v.mean()
    #         # lossD_fake.backward()
    #         # D_G_z1 = output.mean().item()
    #         lossD = lossD_real + lossD_fake
    #         lossD = -1*lossD #mi

    #         # if trainD == True:
    #         lossD.backward()
    #         optimizerD.step()

    #             # running_loss += lossD.item() #TODO

    #         #----------------------------------------------------------------
    #         # Training the Generator
    #         #----------------------------------------------------------------
    #         print('training gen')
    #         netG.zero_grad()

    #         #real data- 
    #         # activns_f2real = netD(data) already there above
    #         # cov? 
            
    #         # z = torch.randn(label.size(0), nz, 1, 1, device=device)
    #         #Warning below one includes -1, excludes 1 (iwant to exclude -1 as well)
    #         # noise = torch.tensor(np.random.default_rng().uniform(-1,1,(BATCH_SIZE,BASIS_SIZE)), dtype=torch.float32).to(device)
    #         # # z = torch.tensor(urng(-1,1,(data.shape))).to(device)
    #         # output = netG(z)
    #         assert fake_pcd.requires_grad == True # .detach() is inplace
    #         #fake_pcd was generated using netG above, but this tijme passing it without detach
    #         outputDG, activns_fake = netD(fake_pcd)
    #         print('fake disc output', outputDG.shape, outputDG)

    #         #TODO test lossG
    #         output_reall, activns_reall = netD(data) # doing again as the graph is done ? after previous backward pass 
    #         #check how to retain graph for the netD

    #         lossG = get_generator_loss(activns_fake,  activns_reall)
    #         lossG.backward()
    #         optimizerG.step()

            
    #         epoch_lossD += lossD.item()        
    #         epoch_lossG += lossG.item()

    #     epoch_lossD /= len(pcdloader) #may not be perfect if last batch is smaller 
    #     epoch_lossG /= len(pcdloader) #(so choose batch size such that 5000%batch_size = 0), 4
        
    #     Gene_losses.append(epoch_lossG)
    #     Disc_losses.append(epoch_lossD)
        
    #     if epoch % (NUM_EPOCHS/5) == 1:
    #         #checkpts
    #         torch.save(netG.state_dict(), f'{GENERATOR_MODEL_DIR}/netG_{epoch}.pth')
    #         torch.save(netD.state_dict(), f'{DISCRIMINATOR_MODEL_DIR}/netD_{epoch}.pth') #no need it seems
    #         # but maybe a trained discriminator can be used for further training of generator(or restarting training of generator)

    #     print('$$$--------------------epoch idx:',epoch,  'lossD:',lossD, 'lossG:',lossG, '--------------------$$$') #

    # plot_losses(Gene_losses, Disc_losses)

    

