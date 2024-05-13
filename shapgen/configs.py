
BASIS_SIZE = 100 #100
NUM_PT_FEATURES = 3

SWAP_K = 10 #10000
ITERS_I = 50 #1000

WIDTH = 1000 #20 #1000  #N in paper
NUMs_SHAPE = 2000 #200  # 5000  #S in paper #5000 toomuch time on cpu should be more than the basis size ( for dim of usvt)
#bottleneck is the svd , need to check svd using cuda skcuda

#for width = 20 (60)  < numsshape 200 - (horizontal matrix)    U: (60, 60) S: (60,) Vt: (60, 200) #why is u 60x60 and not 60x200
#for width = 100(300) > numsshape 20 - (vertical matrix)       U: (300, 20) S: (20,) Vt: (20, 20)

DECREASING_LOSS_PLOTPATH = f'C:/Users/Sairaj Loke/Desktop/Preimage/Preimage_Intern_Task/shapgen/plots/decreasing_pca_loss/decreasing_loss_plot_b{BASIS_SIZE}_nf{NUM_PT_FEATURES}_sk{SWAP_K}_iI{ITERS_I}_w{WIDTH}_ns{NUMs_SHAPE}.png'
POINT_CLOUD_DATA = 'C:/Users/Sairaj Loke/Desktop/Preimage/shapenet-chairs-pcd/'
SORTED_POINTCLOUD_NPY_FILEPATH = f'C:/Users/Sairaj Loke/Desktop/Preimage/Preimage_Intern_Task/shapgen/point_cloud_matrices/sorted_ptcloud_3NxS_b{BASIS_SIZE}_nf{NUM_PT_FEATURES}_sk{10}_iI{100}_w{WIDTH}_ns{5000}.npy' #save value for all pts
OPTIMIZED_SORTED_m_3NxS_NPY_FILEPATH = f'C:/Users/Sairaj Loke/Desktop/Preimage/Preimage_Intern_Task/shapgen/point_cloud_matrices/optimized_sorted_ptcloud_3NxS_b{BASIS_SIZE}_nf{NUM_PT_FEATURES}_sk{SWAP_K}_iI{ITERS_I}_w{WIDTH}_ns{NUMs_SHAPE}.npy'
OPTIMIZED_SORTED_U_FILEPATH = f'C:/Users/Sairaj Loke/Desktop/Preimage/Preimage_Intern_Task/shapgen/point_cloud_matrices/optimized_U_b{BASIS_SIZE}_nf{NUM_PT_FEATURES}_sk{SWAP_K}_iI{ITERS_I}_w{WIDTH}_ns{NUMs_SHAPE}.npy'
'C:/Users/Sairaj Loke/Desktop/Preimage/Preimage_Intern_Task/shapgen/point_cloud_matrices/optimized_sorted_ptcloud_3NxS.npy'
OPTIMIZED_Vt_FILEPATH = f'C:/Users/Sairaj Loke/Desktop/Preimage/Preimage_Intern_Task/shapgen/point_cloud_matrices/optimized_Vt_b{BASIS_SIZE}_nf{NUM_PT_FEATURES}_sk{SWAP_K}_iI{ITERS_I}_w{WIDTH}_ns{NUMs_SHAPE}.npy'
OPTIMIZED_SIGMA_FILEPATH =f'C:/Users/Sairaj Loke/Desktop/Preimage/Preimage_Intern_Task/shapgen/point_cloud_matrices/optimized_sigma_b{BASIS_SIZE}_nf{NUM_PT_FEATURES}_sk{SWAP_K}_iI{ITERS_I}_w{WIDTH}_ns{NUMs_SHAPE}.npy'

TRAINING_DATA_PATH = 'C:/Users/Sairaj Loke/Desktop/Preimage/Preimage_Intern_Task/shapgen/point_cloud_matrices/optimized_sorted_ptcloud_3NxS.npy'
GENERATOR_MODEL_PATH = 'C:/Users/Sairaj Loke/Desktop/Preimage/Preimage_Intern_Task/shapgen/models/generator.pth'
#training params

NUM_EPOCHS = 500
LRD = 0.0001
LRG = 0.0025
REAL_LABEL = 1
FAKE_LABEL = 0
BATCH_SIZE = 3 #(say1)
BETA1 = 0.5
DISC_K_ITERS = 1 # >1 not supported currently in code makesure to keep it 1
