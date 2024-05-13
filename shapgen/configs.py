
BASIS_SIZE = 100 #100
NUM_PT_FEATURES = 3

SWAP_K = 10 #10000
ITERS_I = 50 #1000

WIDTH = 1000 #20 #1000  #N in paper
NUMs_SHAPE = 2000 #200  # 5000  #S in paper #5000 toomuch time on cpu should be more than the basis size ( for dim of usvt)
#bottleneck is the svd , need to check svd using cuda skcuda

#for width = 20 (60)  < numsshape 200 - (horizontal matrix)    U: (60, 60) S: (60,) Vt: (60, 200) #why is u 60x60 and not 60x200
#for width = 100(300) > numsshape 20 - (vertical matrix)       U: (300, 20) S: (20,) Vt: (20, 20)

PLOTPATH_PCA = f'C:/Users/Sairaj Loke/Desktop/Preimage/Preimage_Intern_Task/shapgen/plots/PCA/'
PLOTPATH_TRAINING = f'C:/Users/Sairaj Loke/Desktop/Preimage/Preimage_Intern_Task/shapgen/plots/training/'
PLOTPATH_GENERATED = f'C:/Users/Sairaj Loke/Desktop/Preimage/Preimage_Intern_Task/shapgen/plots/generated/'

DECREASING_LOSS_PLOTPATH = f'C:/Users/Sairaj Loke/Desktop/Preimage/Preimage_Intern_Task/shapgen/plots/decreasing_pca_loss/decreasing_loss_plot_b{BASIS_SIZE}_nf{NUM_PT_FEATURES}_sk{SWAP_K}_iI{ITERS_I}_w{WIDTH}_ns{NUMs_SHAPE}.png'
POINT_CLOUD_DATA = 'C:/Users/Sairaj Loke/Desktop/Preimage/shapenet-chairs-pcd/'
SORTED_POINTCLOUD_NPY_FILEPATH = f'C:/Users/Sairaj Loke/Desktop/Preimage/Preimage_Intern_Task/shapgen/point_cloud_matrices/sorted_ptcloud_3NxS_b{BASIS_SIZE}_nf{NUM_PT_FEATURES}_sk{10}_iI{100}_w{WIDTH}_ns{5000}.npy' #save value for all pts

OPTIMIZED_SORTED_m_3NxS_NPY_FILEPATH = f'C:/Users/Sairaj Loke/Desktop/Preimage/Preimage_Intern_Task/shapgen/point_cloud_matrices/optimized_sorted_ptcloud_3NxS_b{BASIS_SIZE}_nf{NUM_PT_FEATURES}_sk{SWAP_K}_iI{ITERS_I}_w{WIDTH}_ns{NUMs_SHAPE}.npy'
OPTIMIZED_SORTED_U_FILEPATH = f'C:/Users/Sairaj Loke/Desktop/Preimage/Preimage_Intern_Task/shapgen/point_cloud_matrices/optimized_U_b{BASIS_SIZE}_nf{NUM_PT_FEATURES}_sk{SWAP_K}_iI{ITERS_I}_w{WIDTH}_ns{NUMs_SHAPE}.npy'
OPTIMIZED_Vt_FILEPATH = f'C:/Users/Sairaj Loke/Desktop/Preimage/Preimage_Intern_Task/shapgen/point_cloud_matrices/optimized_Vt_b{BASIS_SIZE}_nf{NUM_PT_FEATURES}_sk{SWAP_K}_iI{ITERS_I}_w{WIDTH}_ns{NUMs_SHAPE}.npy'
OPTIMIZED_SIGMA_FILEPATH =f'C:/Users/Sairaj Loke/Desktop/Preimage/Preimage_Intern_Task/shapgen/point_cloud_matrices/optimized_sigma_b{BASIS_SIZE}_nf{NUM_PT_FEATURES}_sk{SWAP_K}_iI{ITERS_I}_w{WIDTH}_ns{NUMs_SHAPE}.npy'

TRAINING_Vt = f'C:/Users/Sairaj Loke/Desktop/Preimage/Preimage_Intern_Task/shapgen/point_cloud_matrices/optimized_Vt_b{BASIS_SIZE}_nf{NUM_PT_FEATURES}_sk{SWAP_K}_iI{ITERS_I}_w{WIDTH}_ns{NUMs_SHAPE}.npy'
INFERENCE_U = OPTIMIZED_SORTED_U_FILEPATH #note, it is dependent on current SWAP_K, ITERS_I, WIDTH, NUMs_SHAPE, so keep them according to the file name to be loaded
INFERENCE_SIGMA = OPTIMIZED_SIGMA_FILEPATH

TRAINING_DATA_PATH = 'C:/Users/Sairaj Loke/Desktop/Preimage/Preimage_Intern_Task/shapgen/train/optimized_Vt_b100_nf3_sk10_iI50_w1000_ns2000.npy'
GENERATOR_MODEL_DIR= 'C:/Users/Sairaj Loke/Desktop/Preimage/Preimage_Intern_Task/shapgen/models/'
DISCRIMINATOR_MODEL_DIR= 'C:/Users/Sairaj Loke/Desktop/Preimage/Preimage_Intern_Task/shapgen/models/'


#training params

NUM_EPOCHS = 10
LRD = 0.0001
LRG = 0.0025
REAL_LABEL = 1
FAKE_LABEL = 0
BATCH_SIZE = 3 #(say1)
BETA1 = 0.5
DISC_K_ITERS = 1 # >1 not supported currently in code makesure to keep it 1
TRAINING_ID = 1

#1 = changed relu in generator was seeing almost zeros in the generated shapes