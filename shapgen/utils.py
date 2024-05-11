
'''
Plotting and drawing functions

'''
import matplotlib.pyplot as plt

def draw_point_cloud(m_3NxS):
    #draw the point cloud
    pass

def plot_error_vs_iters(pca_error_periters_list):
    #plot the error vs iterations
    
    plt.plot(pca_error_periters_list)
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Error vs Iterations')
    
    plt.savefig('plots/error_vs_iters.png')
    plt.show()

    return 