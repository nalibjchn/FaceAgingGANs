## Final Year Research Project (2019/2020):
### Topic: In-depth Study of Generative Adversarial Networks for Face Aging ###

##### Author: Na Li
##### Student Number: 17210325 
##### Supervisor: Guenole Silvestre 
##### UCD School of Computer Science
##### University College Dublin 
##################################### 

The aim of the research project is to study and analysis three exits architectures based on Generative Adversarial Networks (GANs) for face aging tasks in Deep Learning, IPCGANs and CAAE and CycleGANs. The gender and age estimators for evaluating and verifying the synthetic faces which are generated by the three models.

The repo includes all programmes and corresponding to comprehensive structures for running programmes.
The overview instruction below,
1. Environment setup for all programmes

	 (1) Training and Testing Environment
	  - Google Colab notebook:
	     - GPU: Tesla K80 12GB
	  - GPU on Sonic High-Performance Computer (HPC) Cluster of UCD campus.
	     - GPU: "Tesla V100-PCIE-32GB"
<p align="center">
  <img src="infor/GPU_colab.PNG" height="120",width="800"> 
  <img src="infor/GPU_sonic.PNG" height="120",width="800">  
</p>
	 (2) Install 3rd-package dependencies of python (No need for Google Colab)
	 
    In Snoic HPC cluster platform, a virtual environment has to be set up for running programmes by anaconda.
	  
- 3rd-package dependencies

```
     pip python 3.6.8
     pip install tensorflow-gpu==1.14.0
     pip install scipy==1.0.0
     pip install opencv-python==3.3.0.10
     pip install imageio
     pip install scikit-image
     pip install Pillow==5.1.0
     pip install pandas
     pip install -U scikit-learn
     pip install keras==2.2.4
     pip install numpy 

 ```

- Other Libraries (both have pre-installed on Sonic HPC cluster service )
   
   CUDA 10.0 \
   Cudnn 7.0
   
2. Dataset download 
All datasets are shared on Google Drive.

    Link:  https://drive.google.com/drive/folders/1AN4V-cdq0pIUXtXyWBtIcveJI12WZnlh?usp=sharing/DATA.zip [DATA.zip]

3. Project Structures 
        
	- IPCGAN_Face_Aging_5AgeGroups
	- Face_Aging_CAAE_10age
	- CycleGAN
	   - CycleGAN_v1
	   - CycleGAN_v2
	- Age_Gender_Estimators_Experiment_G-Colab
	- Face_Age_Gender_Estimator	
	- DATA
	   - CycleGANs_Paired_TrainingSet
	   - TestSet_FGNET
	   - TrainingSet_CACD2000
	   - ValidationSet_UTKFace

4. Miscellaneous
   
   If the jupyter notebooks (.ipynb) on GitHub cannot load directly, please copy the path to https://nbviewer.jupyter.org/.
