# TopoGAN

TopoGAN is tested under Windows10 and ran on jupyter notebook with pytorch support.
[Paper(https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480120.pdf)]

To run TopoGAN, you will need persistent homology computation that can output cycle information and critical points.
We provide a complied version used in the paper in the form of a python module under folder persis_lib_cpp.
If you are interested to use it as a standalone module, import it with the following lines:
- import sys
- sys.path.insert(0, './persis_lib_cpp')
- from persis_homo_optimal import *
The defination of the interfaces are given in persis_lib_cpp/PersistenceComputer.h

The persistence diagrams of the training images should be pre-computed with the function "FileIO.compute_pd_save"
An example is given in the Main.ipynb.

The hyperparameters of TopoGAN are all defined in file: Archpool.ipynb.
TopoGAN supports easy change of network architecture. The default architectures are DC GANs (index 2 and 3 in Archpool).
TopoGAN works better with ResNet (also provided in Archpool).

Some important hyperparameters are explained here:  
return_settings:  
Basic:  
(1) branch_name: the name of the folder where the logs are saved  
(2) continue_model: if to train model based on previous models  
(3) model_step: if continue_model is true, indicate the model you want to load  

Path:  
(1) save_path: where to save models and log files  
(2) data_path: the folder where there are training images  
(3) pds_path: the folder where there are computed persistence diagrams  
