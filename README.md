# Efficient Spatial Transcriptomic Cell Typing Using Deep Learning and Self-Attention Mechanisms
![figure1_second(ZJL)_final_20240520](https://github.com/coffeei1i/DSCT/assets/97372807/f4bac695-1999-47c3-a8d8-6a6a3ca60f1d)


This repository is the official implementation of DSCT.
## 🚨 environment

To download and start using this project, execute the following commands in your terminal:


```bash
git clone https://github.com/coffeei1i/DSCT.git
cd DSCT-master
conda env create -f environment.yml
conda activate DSCT
python setup.py develop
```


## 🚀 Overview
![figure5_part](https://github.com/user-attachments/assets/f7321537-eb67-48a6-8e35-eead838c0308)




## 🔔 Datasets


### **Tutorial: Human Cortex**

- **sc_CTX_human_ExN**: [Download Link](https://www.dropbox.com/scl/fi/3nl2fmclkdray82z1mxr4/sc_anno_CTX_human.h5ad?rlkey=ouwtemwux3onf68otq95f2xrb&st=70xle7n1&dl=0)
- **st_CTX_human_ExN**: [Download Link](https://www.dropbox.com/scl/fi/i0tly06b8r77avov398a6/CTX_human.h5ad?rlkey=r1gedttub70he0sw064mftnto&st=mjbul9mx&dl=0)

### **Tutorial: Mouse Cortex**

- **sc_CTX_mouse_ExN**: [Download Link](https://www.dropbox.com/scl/fi/t6q6j33afuv7uidh7vi1a/sc_anno_CTX_mouse.h5ad?rlkey=9czlk4n2ar9swgtxr8w0y5u66&st=69h0ckht&dl=0)
- **st_CTX_mouse_ExN**: [Download Link](https://www.dropbox.com/scl/fi/n356h1w3iza8wjo7md6at/CTX_mouse.h5ad?rlkey=hmuhsuxc0rwzmuaxowvcf75an&st=4dtqi7wo&dl=0)

### **Tutorial: HPF Cluster 36/37**
- **sc_HIP_mouse_cluster**: [Download Link](https://www.dropbox.com/scl/fi/e98jvjd34rtt0wzomdk0z/WMB-10Xv3-HPF-DSCT.h5ad?rlkey=ixmgce7fxfqvru3prefnua14k&st=en9h6cab&dl=0)
- **st_HIP_mouse_cluster_36**: [Download Link](https://www.dropbox.com/scl/fi/nfxn73cts2ed24k5mk6b9/HPF_region_36.h5ad?rlkey=8xn8ha159kbgr3iipkl4lb81k&st=1dkq4u8v&dl=0)
- **st_HIP_mouse_cluster_37**: [Download Link](https://www.dropbox.com/scl/fi/kcob2p52r53qp3mfp0qo7/HPF_region_37.h5ad?rlkey=654w5dm4pc4mtuk0sub8kvub8&st=cwhyg15t&dl=0)



## 🤖 Tutorial

see in https://github.com/coffeei1i/DSCT/blob/master/tutorial
## 📝 Important Notes

- If gene names have environment-related duplicates, use the following command to resolve the issue:  
  `sc_obj.var_names = sc_obj.var_names.astype(str)`
- Use `%matplotlib inline` for inline plotting in Jupyter notebooks
- Be aware of potential issues with numpy version 1.23.4

Additionally, we have provided a tutorial for setting up the required environment on the online code platform CodeOcean:  
[https://codeocean.com/capsule/6490820/tree](https://codeocean.com/capsule/6490820/tree)

## 📚 Requirements
To run our code, please install dependency packages.
```bash
python         3.8
torch          1.7.1
numpy          1.23.4  # which is really important
pandas         2.0.3
scanpy         1.9.4
anndata        0.9.2
diopy          0.5.5
cosg           1.0.1
```


## 🤝 About

Should you have any questions, please feel free to contact Mr Xu at yiban@zju.edu.cn.


