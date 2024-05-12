from task1_kdtree import Node, KDTree
import open3d as o3d
import numpy as np  
import time

from configs import WIDTH,NUMs_SHAPE,NUM_PT_FEATURES
from configs import POINT_CLOUD_DATA, SORTED_POINTCLOUD_NPY_FILEPATH

from tqdm import tqdm
'''

time taken for 5k clouds:
final_sorted_pcld shape: (post transpose):  (3000, 5000)
Time taken:  281.01263427734375 ( with print cmds) 

'''
begining = True

def process_point_cloud(pcld_dir,pcld_count):
    
    
    final_sorted_pcld = np.zeros( shape=(1, 3*WIDTH) )
    
    for pcld_index in tqdm(range(1,pcld_count+1)):
        

        pcld_path = pcld_dir + str(pcld_index) + '.pcd'
        pcld = o3d.io.read_point_cloud(pcld_path)
        pcldnp = np.asarray(pcld.points)

        
    
        kdt = KDTree()
        mytree = kdt.build_N_getkdtree(pcld.points) #0 passed in build_kdtree as depth inside this
        print(type(mytree['point'][0]))

        #----debugging
        # global begining
        # if begining:
        print('root after sorting (using tree dict) :' , kdt.getroot()['point']) # the first dividing point
        # begining = False
        #--------

        kdt.inorder_traversal(mytree)
        print(kdt.sorted_pcld.shape)
        print(type(kdt.sorted_pcld), type(kdt.sorted_pcld[0]))
        
        kdt.sorted_pcld = np.delete(kdt.sorted_pcld, [0], 0)
        kdt.sorted_pcld = np.expand_dims(kdt.sorted_pcld,0)
        print('kdt.sorted_pcld.shape :' , kdt.sorted_pcld.shape)

        #as this sorted_pcld is a row vector, it is transposed to a column vector afterwards
        #debug ---------------------------------------------------
        mididx = len(kdt.sorted_pcld[0])//2
        print(mididx)
        print('root from the column vector for S', 
                kdt.sorted_pcld[0][ mididx :mididx+2+1] ) #should be exact same to the one from root
        #----------------------------------------------------------

        final_sorted_pcld = np.append(final_sorted_pcld, kdt.sorted_pcld, axis=0) #append as new row
        print(f'{final_sorted_pcld.shape}')

    final_sorted_pcld = np.delete(final_sorted_pcld, [0], 0)
    print(final_sorted_pcld.shape)
    final_sorted_pcld = final_sorted_pcld.transpose() 
    print('final_sorted_pcld shape: (post transpose): ', final_sorted_pcld.shape)

    with open(SORTED_POINTCLOUD_NPY_FILEPATH, 'wb') as f: #sorted_ptcloud_3NxS.npy
        np.save(f, np.array(final_sorted_pcld))


def testsNviz():
    ply_point_cloud = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud('C:/Users/Sairaj Loke/Desktop/Preimage/shapenet-chairs-pcd/1000.pcd')
    pcdnp = np.asarray(pcd.points)

    print(type(pcd))
    print(type(pcd.points))
    print(type(pcdnp))
    # o3d.visualization.draw_geometries([pcd])
                                    #   zoom=0.3412,
                                    #   front=[0,0,-1.0],
                                #   lookat=[0,0,-5],
                                #   up=[0,0,1])
    # hyperplane that is perpendicular to the corresponding axis.
    print('raw',pcd.points[0])

    kdt = KDTree()
    mytree = kdt.build_kdtree(pcd.points)
    print(type(mytree))
    print('root',mytree['point'])
    print('l',mytree['left']['point'])
    print(mytree['left']['left']['point'])
    print(mytree['left']['right']['point'])

    print('r',mytree['right']['point'])
    print(mytree['right']['left']['point'])
    print(mytree['right']['right']['point'])

    single_shape_sorted_pcd = np.zeros( shape=(1, 1) )

   
    # delete test-----------------------
    a = np.array([1,2,6,4,5])
    a = np.delete(a, [0], 0)
    print(a)


if __name__ == "__main__":

    start = time.time()
    process_point_cloud(POINT_CLOUD_DATA,  NUMs_SHAPE)
    end = time.time()
    print('Time taken: ', end-start)

#-----------------------------------
# mytree_array = kdt.build_kdtree_array(pcdnp)
# print(type(pcdnp[0]))
# print(pcdnp[0])

# mytree_array = kdt.build_kdtree_array(pcdnp)
# print(type(pcdnp[0]))
# print(pcdnp[0])


