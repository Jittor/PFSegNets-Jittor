# PFSegNets-Jittor
![](./images/1)
![](./images/2)
# Introduction
This repo contains the the implementation of CVPR-2021 work: PointFlow: Flowing Semantics Through Points for Aerial Image Segmentation by Jittor
# Install
PFSegNets-Jittor environment requirements:

- System: Linux(e.g. Ubuntu/CentOS/Arch), macOS, or Windows Subsystem of Linux (WSL)
- Python version >= 3.7
- CPU compiler (require at least one of the following)
    - g++ (>=5.4.0)
    - clang (>=8.0)
- GPU compiler (optional)
    - nvcc (>=10.0 for g++ or >=10.2 for clang)
- GPU library: cudnn-dev (recommend tar file installation, reference link)
- Jittor

# DataSet preparation
1. Downloading [iSAID](https://captain-whu.github.io/iSAID/) dataset.
2. Using scripts to crop [iSAID](tools/split_iSAID.py) into patches.
3. Using scripts to convert the original mask of [iSAID](tools/convert_iSAID_mask2graymask.py)
into gray mask for training and evaluating.
4. Finally, you can either change the `config.py` or do the soft link according to the default path in config.

For example, suppose you store your iSAID dataset at `~/username/data/iSAID`, please update the dataset path in `config.py`,
```
__C.DATASET.iSAID_DIR = '~/username/data/iSAID'
``` 
Or, you can link the data path into current folder.

```
mkdir data 
cd data
ln -s your_iSAID_root_data_path iSAID
```

Actually, the order of steps 2 and 3 is interchangeable.

## Pretrained Models

Baidu Pan Link: https://pan.baidu.com/s/1MWzpkI3PwtnEl1LSOyLrLw  4lwf 

Google Drive Link: https://drive.google.com/drive/folders/1C7YESlSnqeoJiR8DWpmD4EVWvwf9rreB?usp=sharing

After downloading the pretrained ResNet, you can either change the model path of `network/resnet_d.py` or do the soft link according to the default path in `network/resnet_d.py`.

For example, 
Suppose you store the pretrained ResNet50 model at `~/username/pretrained_model/resnet50-deep.pth`, please update the 
dataset path in Line315 of `config.py`,
```
model.load_parameters(jt.load("~/username/pretrained_model/resnet50-deep.pth"))
```
Or, you can link the pretrained model path into current folder.
```
mkdir pretrained_models
ln -s your_pretrained_model_path path_to_pretrained_models_folder
```

# Model Checkpoints

<table><thead><tr><th>Dataset</th><th>Backbone</th><th>mIoU</th><th>Model</th></tr></thead><tbody>
<tr><td>iSAID</td><td>ResNet50</td><td>66.3</td><td><a href="https://drive.google.com/file/d/18toZ_wAiOc7jgjzPpUuWVm1D82HSQOny/view?usp=sharing" target="_blank" rel="noopener noreferrer">Google Drive</a>&nbsp;</tr>
</tbody></table>

# Training

To be note that, our models are trained on 4 RTX GPUs with 16GB memory.
 **It is hard to reproduce such best results if you do not have such resources.**
For example, when training PFNet on iSAID dataset:
```bash
sh train_iSAID_pfnet_r50.sh
```

# GAOFEN Contest
- Dataset preparation

  The img folder is the data segmented by visible light sea ice, and sar is the data segmented by the sar fishing ground. All is all data, train and val are training set and validation set divided according to a certain ratio (you can divide it by yourself). Image is the corresponding picture, the suffix is tif (visible light segmentation) or png (sar segmentation), gt is the corresponding annotation file suffix is png
  ```shell
  gaofen
  ├── img
  │   ├── all
  │   │   ├── gt
  │   │   └── image
  │   ├── train
  │   │   ├── gt
  │   │   └── image
  │   └── val
  │       ├── gt
  │       └── image
  └── sar
      ├── all
      │   ├── gt
      │   └── image
      ├── train
      │   ├── gt
      │   └── image
      └── val
          ├── gt
          └── image
  ```

- Modify the data path in the config.py to the correct path (if you only train SAR, you can modify only SAR)
  ```shell
  __C.DATASET.GAOFENSAR ='gaofen/sar'
  __C.DATASET.GAOFENIMG ='gaofen/img'
  ```
- Training:

  To participate in which competition, use the corresponding script to train.
  ```shell
  sh train_gfimg_pfnet_r2n101.sh
  sh train_gfsar_pfnet_r2n101.sh
  ```

  In the sea ice target monitoring competition in the visible light image of Ocean-facing One, the submission got a score of 96.1223.

  In the high-resolution SAR image of the offshore fish farm segmentation data set, the submission got a score of 97.0155.

- Test

  Fill in the model path in run.py as the correct path (Different tracks only need to modify this place, the parameter args.dataset_cls = GAOFENIMG, just to pass the number of categories)
  It is a model test according to the requirements of the competition
  Example of native test
  python run.py gaofen/img/val/image test_img/

  - Package into docker and submit the result according to the official prompts (take the sar track as an example)
    - Package the image (Dockerfile is in the current directory)
      ```
      sudo docker build -t pfnet_sar.
      ```
    - (Optional) Local test
      ```shell
      sudo docker run --rm -it --networknon --gpus all -v gaofen/sar/val/image:/input_path -v test_img:/output_path pfnet_sar
      ```
    - Upload to Alibaba Cloud
      ```shell
      sudo docker tag pfnet_sar registry.cn-beijing.aliyuncs.com/xxx/xxx:pfnet_sar
      sudo docker push registry.cn-beijing.aliyuncs.com/xxx/xxx:pfnet_sar
      ```
# Citation
If you find this repo is helpful to your research. Please consider cite our work.

```
@inproceedings{li2021pointflow,
  title={PointFlow: Flowing Semantics Through Points for Aerial Image Segmentation},
  author={Li, Xiangtai and He, Hao and Li, Xia and Li, Duo and Cheng, Guangliang and Shi, Jianping and Weng, Lubin and Tong, Yunhai and Lin, Zhouchen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4217--4226},
  year={2021}
}
```

# Acknowledgement
This repo is based on official [repo](https://github.com/lxtGH/PFSegNets) by pytorch. 
