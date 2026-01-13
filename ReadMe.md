# GA FundusAutofluorescence (FAF) MedSAM Segmentation

## Repository contents
This repository contains two primary scripts.

1. `GASegTrainV4.py`  
Trains a MedSAM based segmentation model for Geographic Atrophy (GA). Given images and ground truths.

2. `GASegInference.py`  
Runs inference from a CSV, compares predictions to a provided ground truth mask, and saves a visualization per sample that contains:
- original image
- ground truth mask
- predicted mask

A `requirements.txt` file is included for installing dependencies.

## Installation
1. Create and activate a Python environment that matches your system.
2. Install dependencies from `requirements.txt` using pip.

## Input data format

### Training CSVs
Training requires two separate CSV files:
- one training CSV
- one validation CSV

Each CSV must contain at minimum:
- an image path column in png format  
  default name: `image_path`  
  each value should be a filesystem path to the input image
- a mask path column in png format
  default name: `mask_path`  
  each value should be a filesystem path to a binary GA mask for the image

If your CSV uses different column names, override them with:
- `--train_image_col` and `--train_mask_col` for the training CSV
- `--val_image_col` and `--val_mask_col` for the validation CSV

### Inference CSV
Inference requires one CSV file.

The inference CSV must contain at minimum:
- an image path column  
  default name: `image_path`
- a ground truth mask path column  
  default name: `mask_path`

If your CSV uses different column names, override them with:
- `--image_col` and `--mask_col`

## Training

### Command line arguments

Required:
- `--train_data_path`  
  Path to the training dataset CSV
- `--val_data_path`  
  Path to the validation dataset CSV

Data column mapping:
- `--train_image_col`  
  Column name for image files in the training CSV (default: `image_path`)
- `--train_mask_col`  
  Column name for mask files in the training CSV (default: `mask_path`)
- `--val_image_col`  
  Column name for image files in the validation CSV (default: `image_path`)
- `--val_mask_col`  
  Column name for mask files in the validation CSV (default: `mask_path`)

Saving and logging:
- `--model_save_path`  
  Directory to save model checkpoints (default: `./checkpoints`)
- `--tensorboard_log_dir`  
  Directory for TensorBoard logs (default: `./runs`)

Model and hyperparameters:
- `--base_model`  
  MedSAM base model (default: `wanglab/medsam-vit-base`)  
  also supported: `flaviagiammarino/medsam-vit-base`
- `--resume_chkpt`  
  Flag for resuming from checkpoint (default: `False`)
- `--path_to_chkpt`  
  Path to the checkpoint to resume from (default: `None`)
- `--parallel`  
  `True` uses multi GPU `DataParallel`, `False` uses single GPU (default: `True`)
- `--image_size`  
  Two integers `W H` (default: `1024 1024`)  
  SAM resizes images to 1024, so keep at 1024 unless you have a specific reason
- `--batch_size`  
  Batch size (default: `10`, noted as max for L40)
- `--lr`  
  Learning rate (default: `1e-5`)
- `--weight_decay`  
  Weight decay (default: `0`)
- `--num_epochs`  
  Total epochs (default: `100`)
- `--transforms`  
  Apply random transformations to training images (default: `True`)
  
## Limitations
- The model is intended for images that contain GA and has not been trained on no GA images. Performance on images without GA is not reliable.
- The smallest lesion size supported for segmentation is 0.05 mm^2.
  
### Example training command
```bash
python GASegTrainV4.py \
  --train_data_path /path/to/train.csv \
  --val_data_path /path/to/val.csv \
  --train_image_col image_path \
  --train_mask_col mask_path \
  --val_image_col image_path \
  --val_mask_col mask_path \
  --model_save_path ./checkpoints \
  --tensorboard_log_dir ./runs \
  --base_model wanglab/medsam-vit-base \
  --parallel True \
  --image_size 1024 1024 \
  --batch_size 10 \
  --lr 1e-5 \
  --weight_decay 0 \
  --num_epochs 100 \
  --transforms True

### Example inference command
```bash
python GASegInference.py \
  --inference_data_path /path/to/inference.csv \
  --image_col image_path \
  --mask_col mask_path \
  --model_ckpt /path/to/checkpoint.pth \
  --image_size 1024 1024 \
  --batch_size 10 \
  --parallel True \
  --output_save_path ./inference_outputs
'''


