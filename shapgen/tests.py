import numpy as np
import torch
import open3d as o3d

def normtests():

    a = np.array([1,1,1,1])
    a = np.expand_dims(a,0)
    print(a.shape)
    print(np.linalg.norm(a),1)

    b = np.array([-2,-2,-2,-2])
    b = np.expand_dims(b,0)
    print(b.shape)
    print(np.linalg.norm(b))

    c = np.linalg.norm(a-b)
    print(c)


    #dimensionality tests
def shape_tests():
    a = np.array([1,2,3,4,5])   
    print(a.shape)
    a = np.expand_dims(a,0) 
    a.transpose()
    print(a.shape)

    b = np.array([-1,-2,-3,-4,-5])
    b = np.expand_dims(b,0)
    print(b.shape)
    b.transpose()

    c = np.array([0.1,0.2,0.3,0.4,0.5])
    c = np.expand_dims(c,0)
    print(c.shape)
    c.transpose()

    d = np.append(a,b,axis=0)
    d = np.append(d,c,axis=0)

    print(d)
    d = d.transpose()
    print(d)
    print(d.shape)

def matslicing():
    a = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])
    print(a[:,3])

def mean_test():
    a = np.array([[1,2],[3,4],[5,6]])
    b = np.mean(a, axis=1)
    print(b)

def cov_test():
    a = np.array([[1,2],[3,4],[5,6],[7,8],[9,10]])
    # a = np.array([[1,2,3,4],[4,3,2,1], [5,6,7,8],[8,7,6,5],[9,10,11,12])) # say 5 activations
    b = np.array([-1,-2,-3,-4,-5])
    #a,b are obs, their vals are vars

    print(a.shape) #5x2  lastdim*lastdim matrix as lst dims vars in each observation
    c1 = np.cov(a) 
    '''c1
    xmean = (1+3+5+7+9)/5 = 5,  
    ymean = (2+4+6+8+10)/5 = 6  

    c11= (1-5)(1-5)+(3-5)(3-5)+(5-5)(5-5)+(7-5)(7-5)+(9-5)(9-5) = 16+4+0+4+16 = 40/4 = 10
    c12= (1-5)(2-6)+(3-5)(4-6)+(5-5)(6-6)+(7-5)(8-6)+(9-5)(10-6) = 16+4+0+4+16 = 40/4 = 10
    c21= (2-6)(1-5)+(4-6)(3-5)+(6-6)(5-5)+(8-6)(7-5)+(10-6)(9-5) = 16+4+0+4+16 = 40/4 = 10
    c22= (2-6)(2-6)+(4-6)(4-6)+(6-6)(6-6)+(8-6)(8-6)+(10-6)(10-6) = 16+4+0+4+16 = 40/4 = 10
    
    [[4, 4],
    [4, 4]]

    #eg2
    p =  np.array([[1,2],[3,4],[5,6],[7,8],[9,10]])

    xmean = (1+2)/2 = 1.5
    ymean = (3+4)/2 = 3.5
    zmean = (5+6)/2 = 5.5
    wmean = (7+8)/2 = 7.5
    vmean = (9+10)/2 = 9.5

    c2 matrix = 2x2
    c11 = (1-1.5)(1-1.5) + (2-1.5)(2-1.5) = 0.5/(2-1) = 0.5
    c12 = (1-1.5)(3-3.5) + (2-1.5)(4-3.5) = 0.5/(2-1) = 0.5 xwithy
    ......
    so keep activation along column ( 4activns x batch_size)
    cov matrix of size 4x4

    '''

    print(c1)

def batched_cov_test():
    a = np.array([ [[1,2,3,4],[4,3,2,1]],
                   [[5,6,7,8],[8,7,6,5]],
                   [[9,10,11,12],[12,11,10,9]],
                ])

                #batch = 3, activations = 2 (each activation of dim =4 )
    print(a.shape) 

    for i in range(a.shape[0]):
        c = a[i].T
        print(c)
        d = np.cov(c)
        print(d)
        print(c.shape)

def cat():
    a = torch.tensor([1,2,3,4,5])
    b = torch.tensor([6,7,8,9,10])
    c = torch.tensor([11,12,13,14,15])
    a = torch.unsqueeze(a,0).T
    b = torch.unsqueeze(b,0).T
    c = torch.unsqueeze(c,0).T

    print(a.shape)
    e = torch.cat((a,b,c),1)
    print(e.shape)
    print(e)

    f = torch.cov(e)
    g = np.cov(e)
    print(f,g)

def norms():

    a = torch.tensor([[1,2,3,4,5], [1,2,3,4,5]], dtype=torch.float32)
    b = torch.tensor([[0,0,0,0,0], [1,1,1,1,1]] , dtype=torch.float32)
    

    val = torch.norm(a, p=2)
    c = val*val
    print(c)

def pca_recon():

    a = np.array([[1,2,3,4,5], [0,0,0,0,0]], dtype=np.float32).T
    print(a.shape)
    print(np.mean(a, axis=1))
    b = torch.tensor([[0,0,0,0,0], [1,1,1,1,1]] , dtype=np.float32)

def brdctg():
    a = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]]).T
    mu = np.expand_dims(np.mean(a, axis=1), 1)
    b = np.ones((5,1))
    print(b.shape)
    print(a+b)

def torchexp():
    a = torch.tensor([1,2,3,4,5]).T
    b = torch.tensor([[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]]).T
    print(a.shape)
    # a = a.expand(b.size())
    a = a.unsqueeze(1)
    a = a.repeat(1,5)
    print(a,a.shape)

def numpyrepeats():
    a = np.array([1,2,1,4,5]).T
    a = np.expand_dims(a,1)
    a = np.repeat(a, 3, axis=1)
    print(a)
    print((a.sum(axis=0)).sum(axis=0))

def npy2o3d():
    np_points = np.random.rand(100, 3)
    print(type(np_points), np_points.shape)
    # From numpy to Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_points)
    o3d.io.write_point_cloud('C:/Users/Sairaj Loke/Desktop/Preimage/Preimage_Intern_Task/shapgen/eg.pcd', pcd)

if __name__ == '__main__':
    # normtests()
    # mean_test()
    # shape_tests()
    # matslicing()
    # cov_test()
    # cat()
    # batched_cov_test()
    # norms()
    # pca_recon()
    # brdctg()
    # torchexp()
    # numpyrepeats()
    npy2o3d()