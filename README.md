
# MatSparse3D

[PDF](https://openaccess.thecvf.com/content/CVPR2024W/PBDL/papers/Mao_Generating_Material-Aware_3D_Models_from_Sparse_Views_CVPRW_2024_paper.pdf)

This repository contains the code for PBDL 2024 (CVPR Workshop) paper: **Generating Material-Aware 3D Models from Sparse Views**

MatSparse3D introduces a novel approach to generate material-aware 3D models from sparse-view images using generative models and efficient pre-integrated rendering. The output of our method is a relightable model that independently models geometry, material, and lighting, enabling downstream tasks to manipulate these components separately. 

![fig_nv_geo](https://github.com/Sheldonmao/MatSparse3D/assets/22072617/b69d573f-d975-4891-bb18-46d7d7c5a387)

## Install
create envrionment using `mamba` or `conda`. And install additional packages using pip.
``` bash
mamba env create --file environment.yml
pip install -r requirements_git+.txt 
```

Our method use pretrained [Zero123-XL model](https://objaverse.allenai.org/docs/zero123-xl/), the pretrained Zero123-XL weights are required to be download and save in `load/zero123`:
```sh
mkdir load/zero123
cd load/zero123
wget https://zero123.cs.columbia.edu/assets/zero123-xl.ckpt
```

## Datasets
 - **DTU-MVS**: Download the DTU sample set from [DTU-MVS](http://roboimagedata2.compute.dtu.dk/data/MVS/SampleSet.zip). The depth and masking can be estimated by running preprocessing.
 - **sparse-relight**: We provide our generated sparse-relight scenes on [Google Drive](https://drive.google.com/file/d/1F27Ti0pA0CMnUz0ipGypP_gMuZOjUHss/view?usp=sharing). The provided data has estimated depth without the need for preoprocessing

```
 DATA/
 ├── DTU-MVS/
 │   └── SampleSet
 ├── sparse-relight/
     ├── light-probes/
     ├── mesh/
     └── synthesis-images/
          ├── cartooncar/
          ├── gramophone/
          └── ......
```

### Preprocessing (optional)
 [Omnidata](https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/torch) is used for depth and normal prediction. The following ckpts are hardcoded in `preprocess_image.py` and required to be downloaded:
```bash
mkdir load/omnidata
cd load/omnidata
# dowload omnidata_dpt_depth_v2.ckpt
gdown '1Jrh-bRnJEjyMCS7f-WsaFlccfPjJPPHI&confirm=t' 
## optionally download omnidata_dpt_normal_v2.ckpt for normal prediction
# gdown '1wNxVO4vVbDEMEpnAi_jwQObf2MFodcBR&confirm=t' 
```
Preprocessing data, with optional `--use_normal` to estimate the normal if needed. 

``` bash
# e.g. for sparse-relight data
python preprocess.py --scene_dir DATA/sparse-relight/cartooncar

# e.g. for DTU data
python preprocess_DTU.py --scene_dir DATA/DTU-MVS/SampleSet/MVS-Data/Rectified/scan56
```


## Training
Start training by running `launch.py` with `--train` flag. 

- Number of train views can be specified by `data.tran_views`. 
- Config file for `DTU` and `sparse-relight`dataset are available

For more settings, please refer to the config file.
``` bash
# to train the proposed MatSparse3D model on Sparse-Relight dataset
python launch.py --config configs/matsparse3d_sparserelight.yaml --train --gpu 0 data.train_views=5

# to train the proposed MatSparse3D model on DTU dataset
python launch.py --config configs/matsparse3d_DTU.yaml --train --gpu 0 data.train_views=5

# to train the Zero123-n model
python launch.py --config configs/zero123n_sparserelight.yaml --train --gpu 0 data.train_views=5

# to train the nvdiffrec-n model
python launch.py --config configs/nvdiffrec_sparserelight.yaml --train --gpu 0 data.train_views=5
```

## Model Evaluations

### Rendering Interpolated Views

use `--validate` to render interpolated views, an examle is provided in `script_test.sh`
``` bash
# beware to specify the exp_folder
bash script_validate.sh 
```


https://github.com/Sheldonmao/MatSparse3D/assets/22072617/386e8817-36a0-46e5-ac24-9ab1db116db1


### Testing
use `--test` to evaluate on testing views. For sparse-relight dataset, relighting result is also reported. An examle is provided in `script_test.sh`
``` bash
# beware to specify the exp_folder
bash script_test.sh 
```


https://github.com/Sheldonmao/MatSparse3D/assets/22072617/2c93b3d1-c32d-4236-9f93-ed9aac92061d


### Exporting Mesh
use `--export` to export geometry from trained model, an examle is provided in `script_export.sh`
``` bash
# beware to specify the exp_folder
bash script_export.sh 
```

<img src="https://github.com/Sheldonmao/MatSparse3D/assets/22072617/c27f2ac8-83b9-481b-8404-d7fff62e7a22" alt="drawing" width="400"/>


## Credits
Matsparse3D is built on the following open-source projects:

- [Threestudio](https://github.com/threestudio-project/threestudio) A unified framework for 3D content creation
- [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion) for Zero-1-to-3 implementation
- [NeuSPIR](https://github.com/Sheldonmao/NeuSPIR) Learning Relightable Neural Surface using Pre-Integrated Rendering

  
## Citation
If MatSparse3D is relevant to your project, please cite our associated paper:
```
@InProceedings{mao2023matsparse3d,
  author={Mao, Shi and Wu, Chenming and Yi, Ran and Shen, Zhelun and Zhang, Liangjun and Heidrich, Wolfgang},
  title={Generating Material-Aware 3D Models from Sparse Views},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  year={2024},
  month={June},
}
```


