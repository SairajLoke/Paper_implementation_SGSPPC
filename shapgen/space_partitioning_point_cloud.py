from kdtree import Node, KDTree
import open3d as o3d
import numpy as np  

WIDTH = 1000  #N in paper
NUMs_SHAPE = 5000  #S in paper



def process_point_cloud(pcld_dir,pcld_count):
    
    final_sorted_pcld = np.zeros( shape=(1, 3*WIDTH) )
    
    for pcld_index in range(1,pcld_count+1):
        pcld_path = pcld_dir + str(pcld_index) + '.pcd'
        pcld = o3d.io.read_point_cloud(pcld_path)
        pcldnp = np.asarray(pcld.points)
    
        kdt = KDTree()
        mytree = kdt.build_kdtree(pcld.points)
        print(type(mytree['point'][0]))

        kdt.inorder_traversal(mytree)
        print(kdt.sorted_pcld.shape)
        print(type(kdt.sorted_pcld), type(kdt.sorted_pcld[0]))
        
        kdt.sorted_pcld = np.delete(kdt.sorted_pcld, [0], 0)
        kdt.sorted_pcld = np.expand_dims(kdt.sorted_pcld,0)
        print(kdt.sorted_pcld.shape)


        final_sorted_pcld = np.append(final_sorted_pcld, kdt.sorted_pcld, axis=0) #append as new row
        print(f'{final_sorted_pcld.shape}')

    final_sorted_pcld = np.delete(final_sorted_pcld, [0], 0)
    print(final_sorted_pcld.shape)
    final_sorted_pcld = final_sorted_pcld.transpose()
    print('final_sorted_pcld shape: (post transpose): ', final_sorted_pcld.shape)

    with open('sorted_ptcloud_3NxS.npy', 'wb') as f:
        np.save(f, np.array(final_sorted_pcld))



process_point_cloud('C:/Users/Sairaj Loke/Desktop/Preimage/shapenet-chairs-pcd/', 1)



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




#-----------------------------------
# mytree_array = kdt.build_kdtree_array(pcdnp)
# print(type(pcdnp[0]))
# print(pcdnp[0])

# mytree_array = kdt.build_kdtree_array(pcdnp)
# print(type(pcdnp[0]))
# print(pcdnp[0])


