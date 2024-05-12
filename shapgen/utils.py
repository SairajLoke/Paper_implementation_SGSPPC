
'''
Plotting and drawing functions

'''
import matplotlib.pyplot as plt
import open3d as o3d

def draw_point_cloud(pcd):
    o3d.visualization.draw_geometries([pcd])
    #draw the point cloud
    return 

def plot_error_vs_iters(pca_error_periters_list):
    #plot the error vs iterations
    
    plt.plot(pca_error_periters_list)
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Error vs Iterations')
    
    plt.savefig('plots/error_vs_iters.png')
    plt.show()

    return 

def plot_losses(Gene_losses, Disc_losses):
    #plot the losses of the generator and discriminator
    
    plt.plot(Gene_losses, label='Generator Loss')
    plt.plot(Disc_losses, label='Discriminator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Losses vs Epochs')
    plt.legend()
    
    plt.savefig('plots/losses_vs_epochs.png')
    plt.show()

    return
def save_column_matrix_as_pcd(save_path, column_matrix, idx):
    #save the column matrix as a point cloud data file
    column_matrix = column_matrix.reshape(3,-1) # check if it keeps xyz adjacent

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(column_matrix) #column_matrix is a 3xN matrix
    path = save_path + '_' + +str(idx)+'.pcd'
    o3d.io.write_point_cloud(path, pcd)

    return 