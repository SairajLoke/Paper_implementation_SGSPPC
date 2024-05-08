import open3d as o3d
import numpy as np

'''
Code written by Sairaj Loke

'''
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



#my code for kdtree

dims = [0,1,2]


# hyperplane that is perpendicular to the corresponding axis.
print('raw',pcd.points[0])

class Node:
    def __init__(self, point=None, left=None, right=None):
        self.point = point
        self.left = left
        self.right = right


class KDTree :
    def __init__(self):
        self.root = None
        # self.sorted_point_cloud = None

    def build_kdtree(self, points, depth=0):
        n = len(points)
        if n <= 0:
            return None  #empty point cloud

        axis = depth % 3
        sorted_points = sorted(points, key=lambda x: x[axis])

        # https://www.geeksforgeeks.org/python-difference-between-sorted-and-sort/

        
        return {
            'point': sorted_points[n // 2],
            'left': self.build_kdtree(sorted_points[:n // 2], depth + 1),
            'right': self.build_kdtree(sorted_points[n // 2 + 1:], depth + 1)
        }

    # def build_kdtree_array(self, points, depth=0):
    #     n = len(points)
    #     if n <= 0:
    #         return None  #empty point cloud

    #     axis = depth % 3
    #     points.sort()
    #     # https://www.geeksforgeeks.org/python-difference-between-sorted-and-sort/


    #         # 'point': sorted_points[n // 2],
    #     self.build_kdtree_array(points[:n // 2], depth + 1),
    #     self.build_kdtree_array(points[n // 2 + 1:], depth + 1)



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




# mytree_array = kdt.build_kdtree_array(pcdnp)
# print(type(pcdnp[0]))
# print(pcdnp[0])
