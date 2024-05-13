

'''
Aim of this part seems to be to optimize the order of points in matrix 3NxS, 
to calculate a better shape basis of the point cloud.

this means we need to capture more variance by swapping two points

should we focus on smallest variance or largest variance? something like that rather than just swapping two random points


steps

1. random 2 pts i,j (rows multiple of 3)
2. swap
3. if red error keep swapped, else revert swapping
4. 


'''
from numpy.random import default_rng
import numpy as np
import time

from task2_pca import pca_using_svd, skcuda_pca_using_svd

from utils import plot_error_vs_iters, draw_point_cloud
from configs import NUMs_SHAPE, SWAP_K, ITERS_I
from configs import NUM_PT_FEATURES, WIDTH, BASIS_SIZE
from configs import  SORTED_POINTCLOUD_NPY_FILEPATH, OPTIMIZED_SORTED_m_3NxS_NPY_FILEPATH 
from configs import OPTIMIZED_SIGMA_FILEPATH, OPTIMIZED_SORTED_U_FILEPATH
rng = default_rng()

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from tqdm import tqdm   

def pca_reconstruction_error(U,S,Vt, m_3NxS):
    print('calculating_pca_error')
    '''
    avg across all shapes ?

    Here,
    _3NxS : 3N points in S shapes  to convert to 

    mu  (3Nx1) : 
    U   (): 
    Ps  (3Nx1) : shape with index s (s belongs to 1 to 5000)
    Psi (3x1)  : a point in shape s with index i (i belongs to 0 to 2999)
    Psj (3x1)  : a point in shape s with index j (j belongs to 0 to 2999)
     
    if only one shape then
    Ps = m_3NxS[:,s]  # 3Nx1
    mu = np.mean(m_3NxS, axis=1) # 3Nx1
    diff1 = Ps - mu # 3Nx1
    val = np.dot(U,U.T)*(diff1) - diff1 # 3Nx1

    if multiple shapes then
    Ps = m_3NxS # 3NxS
    mu = np.mean(m_3NxS, axis=1) # 3Nx1
    diff1 = Ps - mu # 3NxS check if broadcasting works here
    val = np.dot(U,U.T)*(diff1) - diff1 # 3NxS
    error_mag_for_each_shape = np.dot(val.T,val)  # 1xS
    '''
    # oldproduct = np.array([1,1,1,1,1])
    # newproduct = np.array([1,-1,2,2,0.5])  
    #  # 0 4 1   1 0.25 = 6.25
    #
    # val = newproduct - 
    print('m_3NxS.shape', m_3NxS.shape)
    mu = torch.mean(m_3NxS, dim=1) #calculating at every swap as the order of points is changing or instead i can swap mu as well 
    print('mu.shape', mu.shape)
    # mu = mu.expand(NUM_PT_FEATURES,NUMs_SHAPE).to(device)
    mu = mu.unsqueeze(1)
    mu = mu.repeat(1,NUMs_SHAPE).to(device)
    sum_error = torch.tensor(0.0).to(device)

    
    # where h is an n x I column vector of all Is:
    # for i = I
        # B = X - huT
    
    if 1:
        Ps = m_3NxS.to(device)  # 3Nx1
        # Ps = np.expand_dims(Ps, axis=1)
        print('Ps.shape', Ps.shape)
        # diff1 = Ps - mu # 3Nx1
        # val = np.dot(U,U.T)@(diff1) - diff1
        recon = (U@S@Vt).to(device)
        print('recon', recon.shape)

        #----------------------
        val = recon + mu  - Ps  #followed the recon mentioned in toronto notes
        #----------------------
        print(val.shape)

        # if val.any() >= 0e+00 :  #this should be zero , check
        #     print('val not zero')
        # else:
        #     print('val is zero')
        normval = torch.norm(val, p=2).to(device)
        print('normval', normval.shape)
        sum_error += normval*normval
    else:
        pass
    # for s in tqdm(range(NUMs_SHAPE)):
    # #no need to avg over shapes?

    #     print(m_3NxS.shape) #s
    #     Ps = m_3NxS[:,s]  # 3Nx1
    #     Ps = np.expand_dims(Ps, axis=1)
    #     print('Ps.shape', Ps.shape)
    #     # diff1 = Ps - mu # 3Nx1
    #     # val = np.dot(U,U.T)@(diff1) - diff1
    #     recon = (U@S@Vt).to(device)
    #     print('recon', recon.shape)

    #     #----------------------
    #     val = recon +mu  - Ps  #followed the recon mentioned in toronto notes
    #     #----------------------
    #     # print(val)

    #     # if val.any() >= 0e+00 :  #this should be zero , check
    #     #     print('val not zero')
    #     # else:
    #     #     print('val is zero')
    #     error_mag_for_one_shape = np.dot(val.T,val)  
    #     sum_error += error_mag_for_one_shape


    # 1xS                         
    avg_error = sum_error #/NUMs_SHAPE
    print('sum_error : ', avg_error.shape)
    # print('error_mag_for_each_shape:', error_mag_for_each_shape.shape)
    return avg_error # 1x1


def optimizing_pt_ordering(m_3NxS,swaps_K, iters_I):

    # mu = np.mean(m_3NxS, axis=1) confirmed axis-1 so ### (ai1 + ai2 + ai3)/3 ### ith row
    #  mu will change every time i sawp the points
    # either i should calculate mu every time i swap points 
    # or 
    # i should calculate it once and swap as per m_3NxS is swapped 

    mu = torch.mean(m_3NxS, dim=1).to(device) 
    '''assuming the order in m_3NxS is [x1,y1,z1,x2,y2,z2,...,xN,yN,zN].T (colmn '''
    pca_error_alliters_list = []
    min_avg_pca_error_across_shapes = 1e7 #random large value
    #TODO the avg_pca_error_across_shapes seems high- 128133 (theirs was around 12k)

    optimized_U = None
    optimized_S = None 
    optimized_Vt = None

    for iter_idx in tqdm(range(iters_I)):
        pca_error_allswaps_list = []  
        k_pca_errors = torch.tensor(0.0).to(device)
        # for shape_idx in range(NUMs_SHAPE):
        # i think i should change the order (swap) of points in all shapes at once, 
        # so the order of points is consistent in our matrix, 
        # which is our main goal(to have a consistent global ordering of the point cloud)

        for swap_idx in tqdm(range(swaps_K)):
            

            i,j = (rng.choice(np.arange(0,WIDTH), 2, replace=False))*3
            #0,3,6 ... 2997
            print('random i,j', i,j) 
            print('m_3NxS',m_3NxS.shape)

            #debgu
            if i == j or i%3 != 0 or j%3 != 0 or i<0 or j<0 or i>= NUM_PT_FEATURES*WIDTH  or j>= NUM_PT_FEATURES*WIDTH :
                print(f"error, {i}, {j}")
                return 
            #debug

            Psi = m_3NxS[i:i+3][:].to(device)  #1 random pt across all shapes (to maintain global consistent order)
            Psj = m_3NxS[j:j+3][:].to(device) #
            print('m_3NxS[i:i+3] and j', m_3NxS[i:i+3][:].shape, m_3NxS[j:j+3][:].shape)
            # print('Psi,', Psi, '------------------------' ,'Psj:', Psj ,'------------------------\n')
            print('Psi,Psj:', Psi.shape, Psj.shape )
            m_3NxS[i:i+3][:] = Psj  #TODO check if this works as i want
            m_3NxS[j:j+3][:] = Psi

            print(f'\n----------------------swapped points: {m_3NxS[i:i+3][:].shape}, {m_3NxS[j:j+3][:].shape}')
            

            #m_3NxS has now swapped points i and j
            U,S,Vt = None,None,None

            if device == 'cuda':
                U,S,Vt = skcuda_pca_using_svd(m_3NxS)
            else:
                U,S,Vt = pca_using_svd(m_3NxS)
            # print('U.Ut',np.dot(U,U.T).shape, np.dot(U,U.T ))
            # print('\n-----------------------------------------\n')
                U = torch.from_numpy(U[:,0:BASIS_SIZE]).to(device) #U is 3Nx3N, U[:,0:BASIS_SIZE] is 3Nx100
                S = np.diag(S)
                S = torch.from_numpy(S[0:BASIS_SIZE, 0:BASIS_SIZE]).to(device) #S is 100, S[0:BASIS_SIZE] is 100
                Vt = torch.from_numpy(Vt[0:BASIS_SIZE, 0:NUMs_SHAPE]).to(device) #Vt is 3NxS, Vt[0:BASIS_SIZE] is 100xS
                print('U:', U.get_device(), 'S:', S.get_device(), 'Vt:', Vt.get_device())

            print('U:', U.shape, 'S:', S.shape, 'Vt:', Vt.shape)  #U: (m, 100) S: (100,) Vt: (100, 80)
            print('U.Ut',(U@U.T).shape, U@U.T )
            
            # break
            avg_pca_error_across_shapes = pca_reconstruction_error(U,S,Vt, m_3NxS)

            print('pca_error:',  avg_pca_error_across_shapes.shape, avg_pca_error_across_shapes)

            k_pca_errors += avg_pca_error_across_shapes
            

            if avg_pca_error_across_shapes < min_avg_pca_error_across_shapes :
                min_pca_error = avg_pca_error_across_shapes
                print(f"{iter_idx}, {swap_idx} pca error reduced")
                optimized_U = U
                optimized_S = S
                optimized_Vt = Vt

                #keep changed matrix    
            else:
                #revert back to original
                m_3NxS[i:i+3][:] = Psi
                m_3NxS[j:j+3][:] = Psj
                print(f"{iter_idx}, {swap_idx} pca error not decreased, reverting swap")

        avg_iter_error = k_pca_errors/swaps_K
        pca_error_alliters_list.append(avg_iter_error)

        if iter_idx%10 == 1:
            ploterr = [x.cpu().detach().numpy() for x in pca_error_alliters_list]
            plot_error_vs_iters(ploterr)

        print(pca_error_alliters_list)

    #optimized 
    return m_3NxS, pca_error_alliters_list, optimized_U, optimized_S, optimized_Vt


def get_optimized_pt_orderingNdraw(m_3NxS,swaps_K, iters_I):
    print('inside get_optimized_pt_orderingNdraw')
    print('m_3NxS',type(m_3NxS), m_3NxS.shape)

    m_3NxS, pca_error_periters_list, optimized_U,optimized_S, optimized_Vt = optimizing_pt_ordering(m_3NxS,swaps_K, iters_I)
    print(type(m_3NxS), type(pca_error_periters_list), type(optimized_U))

    pca_error_periters_list = [x.cpu().detach().numpy() for x in pca_error_periters_list]
    optimized_U = optimized_U.cpu().detach().numpy()
    m_3NxS = m_3NxS.cpu().detach().numpy()
    optimized_S = optimized_S.cpu().detach().numpy()

    #draw the optimized point cloud
    # draw(m_3NxS)
    print('Post Optimization point cloud matrix shape: ', m_3NxS.shape)
    print('len of error list: ', len(pca_error_periters_list))
    plot_error_vs_iters(pca_error_periters_list)
    draw_point_cloud(m_3NxS[:,0])
    # np.save('point_cloud_matrices/optimized_U.npy', optimized_U)

    with open(OPTIMIZED_SORTED_U_FILEPATH, 'wb') as f:
       np.save(OPTIMIZED_SORTED_U_FILEPATH, optimized_U)
    with open(OPTIMIZED_SIGMA_FILEPATH, 'wb') as f:
       np.save(OPTIMIZED_SIGMA_FILEPATH, optimized_S)

    return m_3NxS



if __name__ == '__main__':

    optimized_sorted_m_3NxS = None

    

    with open(SORTED_POINTCLOUD_NPY_FILEPATH, 'rb') as f:
        m_3NxS_sorted = np.load(f)
        print('m_3NxS_sorted',m_3NxS_sorted.shape)#, m_3NxS_sorted, '\n-------------------')
        print(type(m_3NxS_sorted))

        #viz some pt clouds and color the pts, to confirm the pts loaded correctly
        draw_point_cloud(m_3NxS_sorted[:,0]) #works

        m_3NxS_sorted = torch.tensor(m_3NxS_sorted).to(device)
        m_3NxS_sorted = m_3NxS_sorted[:3*WIDTH, 0:NUMs_SHAPE].to(device) #for testing
        print('m_3NxS_sorted',m_3NxS_sorted.shape)

        start = time.time()
        optimized_sorted_m_3NxS = get_optimized_pt_orderingNdraw(m_3NxS_sorted,SWAP_K, ITERS_I) #around > 2mins expected
        end = time.time()
        print(f'time taken for sorting ptclds using kdtree: a {3*WIDTH} x {NUMs_SHAPE} matrix: {end-start}')


    print('Optimized sorted point cloud matrix shape: ', optimized_sorted_m_3NxS.shape)
    with open(OPTIMIZED_SORTED_m_3NxS_NPY_FILEPATH, 'wb') as f:
        np.save(f, optimized_sorted_m_3NxS)