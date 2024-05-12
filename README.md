# Preimage_Intern_Task

# Tasks

1. Kdtree
2. PCA
3. Iterative Point Ordering
4. GAN
5. Training


## Setup and running the files

Run the following code: (windows specific venv env activate cmd) 
 ```
 pip -m env preimage_env
 cd preimage_env && Scripts/activate
 git clone https://github.com/SairajLoke/Preimage_Intern_Task
 cd Preimage_Intern_Task/shapgen
 mkdir point_cloud_matrices plots
 pip install -r requirements.txt
 ```
 now your setup is ready! 
 
 ## About the files and folders
 - configs has all the params to tweak and the paths
 - You can viz all plots in plots folders and intermediate sorted npy matrices are saved in point_cloud_matrices folder 
 - ( their (matrices) size is too big, so sharing them files through google drive)
 - 
 
## Inside shapgen folder - 
 To do a kdtree based space partitioning of given point clouds
 `
 python task1_space_partitioning_point_cloud.py
 `
PCA using SVD code can be found in `task2_pca.py`
 To do further optimization over the order of points run
 `
 python task3_optimizing_pt_ordering.py
 `
 
To train a GAN run
`
python task5_training.py
`
To run inference using pretrained gan weigths
`python shapegeneration.py`

references

for theory:
### PCD
https://pointclouds.org/documentation/tutorials/pcd_file_format.html
https://www.open3d.org/

### Kdtree:
https://youtu.be/Ash4Q06ZcHU?si=ifr9WVh2NGxC7A4f
https://pcl.readthedocs.io/projects/tutorials/en/master/kdtree_search.html

### PCA and SVD
https://youtu.be/XwTW_YA3HG0?si=uAE2fRClRvlPbdTw
https://www.cs.toronto.edu/~jepson/csc420/notes/introSVD.pdf 

### GANs
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
https://arxiv.org/pdf/1406.2661 (GANs 2014 pseudo code)

### Others

Pytorch , numpy  , Open3d Documentation .
stackoverflow ( for quick referencing queries like matrix to open3d pt cloud, etc) 
