
import torch
import numpy as np
from configs import REAL_LABEL



class ShapeDataset(torch.utils.data.Dataset):
    def __init__(self,training_data_path):
        # super(ShapeDataset, self).__init__() torch Dataset doesnt have a init of its own

        #optimized and sorted point cloud
        vt_data = np.load(training_data_path)
        print("Loaded pcd_data :" , vt_data.shape) # should be BASIS_SIZE x NUMs_SHAPE = eg. 100x2000

        self.vt_data = torch.tensor(vt_data, dtype=torch.float32)#TODO check precision
        
        print(type(vt_data))
        # print(type(pcd_data.points))
        # print(type(pcdnp))
        # o3d.visualization.draw_geometries([pcd])
                                        #   zoom=0.3412,
                                        #   front=[0,0,-1.0],
                                    #   lookat=[0,0,-5],
                                    #   up=[0,0,1])
        # hyperplane that is perpendicular to the corresponding axis.

    def __len__(self):
        return self.vt_data.shape[1]

    def __getitem__(self, idx):
        
        # a column of Vt as the shape coefficients
        return self.vt_data[:,idx] 

