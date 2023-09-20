# Towards Faster Training of Global Covariance Pooling Networks by Iterative Matrix Square Root Normalization

## Introduction
Deep convolutional neural networks (ConvNets) have made significant progress in recent years, achieving recognition accuracy surpassing human capabilities in large-scale object recognition. This paper focuses on the global covariance pooling in ConvNets, which has shown impressive improvement over the classical first-order pooling. The main challenge addressed in this paper is the inefficiency in training due to the heavy dependence on eigendecomposition (EIG) or singular value decomposition (SVD). The authors propose an iterative matrix square root normalization method for faster end-to-end training of global covariance pooling 3networks.

At the core of iSQRT-COV is a meta-layer with loop-embedded directed graph structure, specifically designed for ensuring both convergences of Newton-Schulz iteration and performance of global covariance pooling networks. 
The meta-layer consists of three consecutive structured layers, performing pre-normalization, coupled matrix iteration, and postcompensation, respectively.


![](https://markdown.liuchengtu.com/work/uploads/upload_e3d8507caa72fab8368ac263e9c0c8d5.png)

## Main Results
|Method           | Acc@1(%) | #Params.(M) | FLOPs(G) | Checkpoint                                                          |
| ------------------ | ----- | ------- | ----- | ------------------------------------------------------------ |
| ResNet-50   |  76.07 |  25.6   |   3.86  |               |
| ResNet-50+ISQRT-COV(Ours)   | 0  |   0  |  0   |[Download](https://drive.google.com/file/d/1PBy8evHi-xiJHiTWgqrUs8jTH58hJM2n/view?usp=share_link)|

## Usage

### Environments
●OS：18.04  
●CUDA：11.6  
●Toolkit：mindspore1.9  
●GPU:GTX 3090 


### Install
●First, Install the driver of NVIDIA  
●Then, Install the driver of CUDA  
●Last, Install cudnn

create virtual enviroment mindspore
conda create -n mindspore python=3.7.5 -y
conda activate mindspore
CUDA 10.1 
```bash
conda install mindspore-gpu cudatoolkit=10.1 -c mindspore -c conda-forge
```
CUDA 11.1 
```bash
conda install mindspore-gpu cudatoolkit=11.1 -c mindspore -c conda-forge
```
validataion 
```bash
python -c "import mindspore;mindspore.run_check()"
```

### Data preparation
Download and extract ImageNet train and val images from http://image-net.org/. (https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val/` folder respectively:


```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```
### Evaluation
To evaluate a pre-trained model on ImageNet val with GPUs run:

```
CUDA_VISIBLE_DEVICES={device_ids}  python eval.py --data_path={IMAGENET_PATH} --checkpoint_file_path={CHECKPOINT_PATH} --device_target="GPU" --config_path={CONFIG_FILE} &> log &
```

### Training

#### Train with ResNet

You can run the `main.py` to train as follow:

```
mpirun --allow-run-as-root -n {RANK_SIZE} --output-filename log_output --merge-stderr-to-stdout python train.py  --config_path={CONFIG_FILE} --run_distribute=True --device_num={DEVICE_NUM} --device_target="GPU" --data_path={IMAGENET_PATH}  --output_path './output' &> log &
```
For example:

```bash
mpirun --allow-run-as-root -n 4 --output-filename log_output --merge-stderr-to-stdout python train.py  --config_path="./config/resnet50_imagenet2012_config.yaml" --run_distribute=True --device_num=4 --device_target="GPU" --data_path=./imagenet --output_path './output' &> log &
```

## Acknowledgement
The work was supported by the National Natural Science Foundation of China (No. 61471082). Peihua Li is the corresponding author.
