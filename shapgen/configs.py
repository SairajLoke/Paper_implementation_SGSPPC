
BASIS_SIZE = 100 #100
NUM_PT_FEATURES = 3

SWAP_K = 4 #10000
ITERS_I = 10 #1000

WIDTH = 500 #20 #1000  #N in paper
NUMs_SHAPE = 120 #200  # 5000  #S in paper

#for width = 20 (60)  < numsshape 200 - (horizontal matrix)    U: (60, 60) S: (60,) Vt: (60, 200) #why is u 60x60 and not 60x200
#for width = 100(300) > numsshape 20 - (vertical matrix)       U: (300, 20) S: (20,) Vt: (20, 20)

POINT_CLOUD_DATA = 'C:/Users/Sairaj Loke/Desktop/Preimage/shapenet-chairs-pcd/'
SORTED_POINTCLOUD_NPY_FILEPATH = 'C:/Users/Sairaj Loke/Desktop/Preimage/Preimage_Intern_Task/shapgen/point_cloud_matrices/sorted_ptcloud_3NxS.npy'
OPTIMIZED_SORTED_m_3NxS_NPY_FILEPATH = f'C:/Users/Sairaj Loke/Desktop/Preimage/Preimage_Intern_Task/shapgen/point_cloud_matrices/optimized_sorted_ptcloud_3NxS_{BASIS_SIZE}_{NUMs_SHAPE}.npy'


TRAINING_DATA_PATH = 'C:/Users/Sairaj Loke/Desktop/Preimage/Preimage_Intern_Task/shapgen/point_cloud_matrices/optimized_sorted_ptcloud_3NxS.npy'

#training params

NUM_EPOCHS = 500
LRD = 0.0001
LRG = 0.0025
REAL_LABEL = 1
FAKE_LABEL = 0
BATCH_SIZE = 3 #(say1)
BETA1 = 0.5
DISC_K_ITERS = 1 # >1 not supported currently in code makesure to keep it 1
