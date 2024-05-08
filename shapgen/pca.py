

import open3d as o3d
import numpy as np

filepath = 'C:/Users/Sairaj Loke/Desktop/Preimage/Preimage_Intern_Task/shapgen/sorted_ptcloud_3NxS.npy'

with open(filepath, 'rb') as f:
    b = np.load(f)
    print(b, b.shape)
    print(type(b))

    