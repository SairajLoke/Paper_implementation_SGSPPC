import numpy as np

'''
Code written by Sairaj Loke

'''
NUM_PT_FEATURES = 3




#my code for kdtree--------------------------------------------
class Node:
    def __init__(self, point=None, left=None, right=None):
        self.point = point
        self.left = left
        self.right = right


class KDTree :
    def __init__(self):
        self.root = None
        self.dims = [0,1,2]
        self.sorted_pcld = np.zeros( shape=(1, 1) )

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

        
    def inorder_traversal(self,node, depth=0):
        #to add new nodes as cols (ie. the sorted pcld is 1xN form always)
        if node is not None :
            self.inorder_traversal(node['left'], depth+1)
            self.sorted_pcld = np.append(self.sorted_pcld, [node['point']]) #axis nt specified, so flattened and concat 
            self.inorder_traversal(node['right'],depth+1)                   # https://numpy.org/doc/stable/reference/generated/numpy.append.html
            

        
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


