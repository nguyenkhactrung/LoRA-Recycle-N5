## Introduction
【CVPR 2025】The PyTorch implementation of

>Unlocking Tuning-Free Few-Shot Adaptability in Visual Foundation Models by Recycling Pre-Tuned LoRAs

<!-- <p align="center">
  <img src="fig/motivation.png" alt="motivation" width="400"><br>
</p>

**Motivation of LoRA Recycle**: Thanks to the modularity of LoRA, users can upload locally tuned LoRAs to public repositories without exposing original training data.
  LoRA Recycle distills a meta-LoRA from these LoRAs without needing their original training data. The VFM, once equipped with the meta-LoRA, is empowered to solve new few-shot tasks in a single forward pass without further fine-tuning.

<p align="center">
  <img src="fig/pipeline.png" alt="motivation" width="400"><br>
</p>

**Pipeline of LoRA Recycle**. (i) (Pink Path) We generate task-specific synthetic data from the pre-tuned LoRA via LoRA Inversion. The input data (attached with the fire in the left corner) is initialized as Gaussian noise and iteratively optimized. The synthetic data is then used to construct a meta-training task with one support set and one query set. (ii) (Black Path) We meta-train the meta-LoRA (attached with the fire in the middle) on a  wide range of pre-tuned LoRAs by minimizing the meta-learning objective, explicitly teaching it how to adapt without fine-tuning.

<p align="center">
  <img src="fig/double.png" alt="motivation" width="400"><br>
</p>

**Double-Efficient Mechanism**. (Left: Efficient Data-Generation) During the data-generation stage, token pruning is performed in the hidden layers by removing unimportant tokens based on self-attention weights, accelerating both forward and backward computations for reverse engineering. (Right: Efficient Meta-Training) To select the most informative tokens from the synthetic data for the following meta-training, we construct a mask by setting values of 1 at the positions of remaining tokens and 0 elsewhere. We multiply the mask with the synthetic image to create a masked image. We then exclusively use the unmasked tokens for meta-training. This selective use of sparse tokens significantly accelerates meta-training, while maintaining or even  improving performance by reducing noise from the synthetic data. -->

## Requirements

```bash
pip install -r requirements.txt
```

## Datasets
1) Download

   In our paper, we use eight datasets.
For In-Domain setting, we use CIFAR-FS, MiniImageNet, VGG-Flower and CUB-200-2011.
For Cross-Domain setting, we use ChestX, ISIC, EuroSAT and CropDiseases.

2) Generate split csv files of meta-training and meta-testing subsets

   Reset the DATA_PATH (the path to the downloaded dataset) and SPLIT_PATH (the path to save the generated csv file) in each file under ./write_file.
    
    Run the following scripts to generate the split csv files of meta-training and meta-testing subsets.
    ```bash
    python ./write_file/write_cifarfs_filelist.py
    python ./write_file/write_miniimagenet_filelist.py
    python ./write_file/write_flower_filelist.py
    python ./write_file/write_cub_filelist.py
    python ./write_file/write_cropdiseases_filelist.py
    python ./write_file/write_eurosat_filelist.py
    python ./write_file/write_isic_filelist.py
    python ./write_file/write_chest_filelist.py
    ```

3) Reset the SPLIT_PATH (the path to the generated split csv files) in each file under ./dataset.
## Quick Start
- In-Domain Setting.
  ```bash
  python main.py \
  --multigpu 6 \
  --gpu 0 \
  --dataset flower \
  --testdataset flower \
  --val_interval 100 \
  --backbone base_clip_16 \
  --resolution 224 \
  --method pre_dfmeta_ft \
  --episode_batch 1 \
  --way_train 5 \
  --num_sup_train 1 \
  --num_qur_train 15 \
  --way_test 5 \
  --num_sup_test 1 \
  --num_qur_test 15 \
  --episode_train 240000 \
  --episode_test 100 \
  --outer_lr 1e-3 \
  --rank 4 \
  --synthesizer inversion \
  --prune_layer -1 \
  --prune_ratio 0.0 \
  --mask_ratio -1 \
  --pre_datapool_path you_should_input_the_path_to_pre-inverted_data
  
  key arguments:
  --dataset: meta-training on inverted data from flower/cifar100/miniimagenet/cub
  --testdataset: meta-testing on real data from flower/cifar100/miniimagenet/cub
  --val_interval: validation interval (iterations)
  --backbone: base_clip_16/base_clip_32, 16/32 indicates the patch size
  --resolution: resolution of images
  --episode_batch: batch size of tasks at one iteration
  --way_train/test: the number of classes during meta-training/testing
  --num_sup_train/test: the number of shot during meta-training/testing
  --episode_train: total number of iterations of meta-training
  --episode_test: number of meta-testing tasks
  --outer_lr: learning rate
  --rank: rank of lora
  --prune_layer: the layers to perform token pruning, 
                 -1 indicates no token pruning
                 l1, l2, ..., ln: perform token pruning at l1, l2, ..., ln
  --prune_ratio: the proportion of tokens to be pruned, relative to the current remaining tokens
                0: indicates no token pruning
                r1 r2 ... rn: progressively prune a fraction (r1, r2, ..., rn) of patches at layers (l1, l2, ..., ln), respectively
  --mask_ratio:
                -1: automatically mask the inverted data based on the positions of remaining tokens
                x: mask extra remaining tokens after the last layer
  --pre_datapool_path: path to the pre-inverted data
  ```
- Cross-Domain Setting.
  ```bash
  python main.py \
  --multigpu 6 \
  --gpu 0 \
  --dataset mix \
  --testdataset cropdiseases \
  --val_interval 100 \
  --backbone base_clip_16 \
  --resolution 224 \
  --method pre_dfmeta_ft \
  --episode_batch 1 \
  --way_train 5 \
  --num_sup_train 1 \
  --num_qur_train 15 \
  --way_test 5 \
  --num_sup_test 1 \
  --num_qur_test 15 \
  --episode_train 240000 \
  --episode_test 100 \
  --outer_lr 1e-3 \
  --rank 4 \
  --synthesizer inversion \
  --prune_layer -1 \
  --prune_ratio 0.0 \
  --mask_ratio -1 \
  --pre_datapool_path you_should_input_the_path_to_pre-inverted_data
  
  key arguments:
  --dataset: meta-training on inverted data from flower+cifar100+miniimagenet+cub
  --testdataset: meta-testing on real data from cropdiseases/eurosat/isic/chest
  --val_interval: validation interval (iterations)
  --backbone: base_clip_16/base_clip_32, 16/32 indicates the patch size
  --resolution: resolution of images
  --episode_batch: batch size of tasks at one iteration
  --way_train/test: the number of classes during meta-training/testing
  --num_sup_train/test: the number of shot during meta-training/testing
  --episode_train: total number of iterations of meta-training
  --episode_test: number of meta-testing tasks
  --outer_lr: learning rate
  --rank: rank of lora
  --prune_layer: the layers to perform token pruning, 
                 -1 indicates no token pruning
                 l1, l2, ..., ln: perform token pruning at l1, l2, ..., ln
  --prune_ratio: the proportion of tokens to be pruned, relative to the current remaining tokens
                0: indicates no token pruning
                r1 r2 ... rn: progressively prune a fraction (r1, r2, ..., rn) of patches at layers (l1, l2, ..., ln), respectively
  --mask_ratio:
                -1: automatically mask the inverted data based on the positions of remaining tokens
                x: mask extra remaining tokens after the last layer
  --pre_datapool_path: path to the pre-inverted data
  ```
## Pre-Invert Data & Visualize
  ```bash
  python main.py \
  --multigpu 6 \
  --gpu 0 \
  --dataset flower \
  --testdataset flower \
  --val_interval 100 \
  --backbone base_clip_16 \
  --resolution 224 \
  --method pre_dfmeta_ft \
  --episode_batch 1 \
  --way_train 5 \
  --num_sup_train 5 \
  --num_qur_train 15 \
  --way_test 5 \
  --num_sup_test 5 \
  --num_qur_test 15 \
  --episode_train 240000 \
  --episode_test 100 \
  --outer_lr 1e-3 \
  --rank 4 \
  --synthesizer inversion \
  --prune_layer -1 \
  --prune_ratio 0.0 \
  --mask_ratio -1 \
  --pre_datapool_path you_should_input_the_path_to_pre-inverted_data \
  --preGenerate
  
  key arguments:
  --preGenerate: set True for inversion
  --pre_datapool_path you should input_the path to save the inverted data
  other arguments are the same as mentioned above
  ```





## Pre-trained ViTs

You can download the pre-trained ViT from the following public link.

```bash
https://huggingface.co/openai/clip-vit-base-patch16
https://huggingface.co/openai/clip-vit-base-patch32
````

## Pre-Tuned LoRAs


You can put the pre-trained LoRA into the following folder.

```bash
./lorahub
```

## Pre-Inverted Data

Please connect me if you need the examples of inverted synthetic images.

![alt text](fig/examples.png)
<!-- We will provide data inverted from LoRAs pre-tuned on different datasets in the following directory (including unmasked and 25%/50%/75%-masked versions). -->
<!-- 
```bash
```bash
./pre_datapool
``` -->

## 