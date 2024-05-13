

import open3d as o3d
import numpy as np
from numpy import linalg as la

from configs import BASIS_SIZE
# U should be calculated on the entire dataset right?
import torch
# from skcuda import linalg as skcudala

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def pca_old():
    with open(filepath, 'rb') as f:
        b = np.load(f)
        print(b, b.shape)
        print(type(b))

        # scaling=StandardScaler()# Use fit and transform method 
        # scaling.fit(pointcloud)

        pca.fit(b) #pretty fast no need to time
        print(pca.explained_variance_ratio_)
        print(pca.singular_values_)


def pca_using_svd(m_3NxS):

    #do some preprocessing ( like centering of matrix ,
    #  check what exactly)
    #-------------------------------------------
    if device == 'cuda':
        m_3NxS = m_3NxS.clone().detach().to('cpu').numpy()

    U, S, Vt = la.svd(m_3NxS, full_matrices=True)
    #expected shapes?  seems U V are switched? not sure

    # V = Vt.T 
    print('U:', U.shape, 'S:', S.shape, 'Vt:', Vt.shape)
    # return U[:,0:BASIS_SIZE],S[0:BASIS_SIZE,0:BASIS_SIZE],V[0:BASIS_SIZE, 0:BASIS_SIZE]
    return U,S,Vt


def skcuda_pca_using_svd(m_3NxS):

    #do some preprocessing ( like centering of matrix ,
    #  check what exactly)
    #-------------------------------------------
    if device == 'cuda':
        m_3NxS = m_3NxS.clone().detach().to('cpu').numpy()

    U, S, Vt = skcudala.svd(m_3NxS, full_matrices=True)
    #expected shapes?  seems U V are switched? not sure

    # V = Vt.T 
    print('U:', U.shape, 'S:', S.shape, 'Vt:', Vt.shape)
    # return U[:,0:BASIS_SIZE],S[0:BASIS_SIZE,0:BASIS_SIZE],V[0:BASIS_SIZE, 0:BASIS_SIZE]
    return U,S,Vt