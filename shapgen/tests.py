import numpy as np


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

def mean_test():
    a = np.array([[1,2],[3,4],[5,6]])
    b = np.mean(a, axis=1)
    print(b)
if __name__ == '__main__':
    # normtests()
    mean_test()
    # shape_tests()