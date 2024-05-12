
import torch
import numpy as np
from configs import REAL_LABEL



class ShapeDataset(torch.utils.data.Dataset):
    def __init__(self,training_data_path):
        # super(ShapeDataset, self).__init__()

        pcd_data = np.load(training_data_path)
        print("Loaded pcd_data :" , pcd_data.shape)
        self.pcd_data = torch.tensor(pcd_data, dtype=torch.float32) #???
        # self.batch_size = batch_size
        # self.shuffle = shuffle

        
        # ply_point_cloud = o3d.data.PLYPointCloud()
        # pcd = o3d.io.read_point_cloud('C:/Users/Sairaj Loke/Desktop/Preimage/shapenet-chairs-pcd/1000.pcd')
        # pcdnp = np.asarray(pcd.points)

        print(type(pcd_data))
        # print(type(pcd_data.points))
        # print(type(pcdnp))
        # o3d.visualization.draw_geometries([pcd])
                                        #   zoom=0.3412,
                                        #   front=[0,0,-1.0],
                                    #   lookat=[0,0,-5],
                                    #   up=[0,0,1])
        # hyperplane that is perpendicular to the corresponding axis.

    def __len__(self):
        return len(self.pcd_data) # self.pcd_data.shape[0]

    def __getitem__(self, idx):
        
        return self.pcd_data[:,idx]  #, REAL_LABEL

