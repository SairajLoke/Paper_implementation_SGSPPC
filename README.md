# Paper implementation 
Paper - https://arxiv.org/pdf/1707.06267

Below is the implementation of the same
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
 git clone https://github.com/SairajLoke/Preimage_Tasks
 cd Preimage_Tasks/shapgen
 pip install -r requirements.txt
 ```
create additional directionaries namely
- models
- point_cloud_matrices  ( to keep intermediate U Vt Sigma & PointCloud data in matrix format)
- train (required while training)

 now your setup is ready! 

 ## To quickly run inference run -
 `python .\task6_inference.py`
 
 mention appropriate 
 - training id and epochs associated with the model to load
 - paths for U and Sigma matrices stored as npy files (shared on google drive)
 - path for loading generator

you can expect a point cloud window popup, along with the correspond pcd stored in inference folder
something like this
![generated_tid1_e8](https://github.com/SairajLoke/Preimage_Tasks/assets/104747561/146f6422-ce28-4468-bb6c-961a08e2d857)

 
 ## About the files and folders

 ### files - 
 - configs:  has all the params to tweak and the paths
 - tests: tests i performed on various lib methods during the entire process
- notes: some pts
 - colab configs and colab notebook were used to use colab specs in a better way
 - other task related files mentioned below
   
### folders
 - You can viz all plots in *plots* folders
 - *latest* : impt results to look at from the different tasks
 - *point_cloud_matrices* : Intermediate sorted npy matrices are saved in folder 
                            ( their (matrices) size is too big, so sharing them files through google drive)
 - *models* : has all the checkpoints saved ( as given in the configs - constant: GENERATOR_MODEL_DIR
 - *train* : not necessary (just to store the Vt matrices used in training), can be changed by paths in configs
 - *inference* : has saved pcd files of generated ptclds


## Inside shapgen folder - 
 To do a kdtree based space partitioning of given point clouds
 
 `
 python task1_space_partitioning_point_cloud.py
 `

In this the order is inorder traversal of the kdtree.
 
PCA using SVD code can be found in `task2_pca.py` 

To do further optimization over the order of points run
 `
 python task3_optimizing_pt_ordering.py
 `
(the torch version of the file is also given, but wasnt that useful)

I think we can further optimize this by swapping a set of adjacent points with another adjacent set of points rather than just 2.

The GAN has been implemented using 4 Linear each in disc + gen,
the architecture is a bit modified (compared to the paper),
specifically - 
 tanh as activation in generator (instead of relu), to generate negative vals as well

To train a GAN run
`
python task5_training_new.py
`
Training uses the Discrimator's activation based loss for generator and vanilla GAN loss for discriminator.
Moreover Discriminator trains only if accuracy of Disc < 0.8

To run inference using pretrained gan weigths
`python task6_inference.py`


## Some selected results from PCA

Sorting using the algo mentioned ( used a matrix representation for calculating the recon error for all shapes rather than iterating over all the vertical pt clouds separately, this considerably reduced the pca recon error calc time ( around 3times less for particular configs)

for 2000 chair shapes ( 5k was taking too long) in the SVD section.
10 swaps per iterations for 50 iterations
<img src="https://github.com/SairajLoke/Preimage_Tasks/assets/104747561/c231f0f3-1e60-4e35-8685-8356a69f3fb2" alt="PCA recon error" width="300" height="250"/>


## GAN and training 
The Trained GAN models (Generator and Discriminator(not needed for inference) ) can be found in models directory -

The training losses can be seen as follows : (saved in train plots)

<img src="https://github.com/SairajLoke/Preimage_Tasks/assets/104747561/f0d193cc-b033-46b9-925a-012c7f249901" alt="Gerror" width="300" height="250"/>
<img src="https://github.com/SairajLoke/Preimage_Tasks/assets/104747561/ad2a2a06-9be5-437b-9667-3d105d9575cb" alt="Derror" width="300" height="250"/>


## References

papers
- https://arxiv.org/pdf/1406.2661 (2014 GAN)
- https://arxiv.org/pdf/1606.03498 (the improved generator loss)

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


