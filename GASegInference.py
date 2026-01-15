import argparse
import os
import random
import re
import sys
from statistics import mean
import cv2
import matplotlib.pyplot as plt
import monai
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from datasets import Dataset as HFDataset
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from scipy import ndimage
from scipy.stats import bootstrap
from torch.nn import DataParallel
from torch.nn.functional import interpolate, normalize, threshold
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import default_collate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import SamModel, SamProcessor
import albumentations as A
from monai.metrics import DiceMetric

COMPARISONS_DIR = 'comparisons'
PREDICTIONS_DIR = 'predictions'

def entire_image_bounding_box(image,size):
    H, W = size
    bbox = [0, 0, W, H]#np.array([0, 0, W, H])
    return bbox
def to_rgb(img):
    '''
    Likely this channel transformation is unecessary, 
    but I want to ensure everything is RGB for SAM which assumes RGB
    '''
    if img is None:
        return None
    if img.ndim == 2: # grayscale to RGB
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
class MedSAMDataset(TorchDataset):
    def __init__(self, dataset, processor, image_size):
        self.dataset = dataset
        self.processor = processor
        self.size = image_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx):
        item = self.dataset[idx]
         #read img on and resize
        img_raw = cv2.imread(item["image"],cv2.IMREAD_UNCHANGED)
        img_raw = to_rgb(img_raw)
        img_raw = cv2.resize(img_raw, self.size, interpolation=cv2.INTER_LINEAR)
        #read mask on and resize
        mask_raw = cv2.imread(item["label"],cv2.IMREAD_GRAYSCALE)
        mask_raw = cv2.resize(mask_raw, self.size, interpolation=cv2.INTER_NEAREST)
        image = np.array(img_raw, dtype=np.uint8)
        ground_truth_mask = np.array(mask_raw,dtype=np.uint8)
        
        #obtain bounding box for model prompt
        prompt = entire_image_bounding_box(ground_truth_mask,self.size)#get size of mask which is sam size as image
        #prepare for model
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors = "pt")
        #remove dimensions of the batch
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}
        #add ground truth seg
        inputs["ground_truth_mask"] = ground_truth_mask
        inputs["raw_image"] = image
        inputs['filename'] = item['image']

        return inputs
        
def LoadData(img_paths,gt_paths,processor,image_size=(1024,1024),batch_size=10):
    '''
    With imread unchanged reading in images with 4 channels as a BGR
    need 3 channel RGB for sam
    For masks need to use inter_nearest to ensure 0/1 does not get blended into gray
    '''
    datadict = {"image": img_paths,"label": gt_paths}
    data = HFDataset.from_dict(datadict)
    dataset = MedSAMDataset(dataset=data, processor=processor, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
            pin_memory=True)
    return dataloader

def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            print("BrokenStateDict")  # already a raw state_dict
    else:
        state = ckpt

    # strip common wrappers
    def strip_prefixes(name):
        for p in ("module.", "model."):
            if name.startswith(p):
                return name[len(p):]
        return name

    return {strip_prefixes(k): v for k, v in state.items()}

def load_any_ckpt(model, path, device="cpu", strict=False):
    '''
    Needed because model weight format is stored different if DataParallel is used v.s. no Parallel.
    '''
    try:
        ckpt = torch.load(path, map_location=device)  # PyTorch 2.6 defaults to weights_only=True
    except Exception:
        ckpt = torch.load(path, map_location=device, weights_only=False)  # trusted only

    state = _extract_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing or unexpected:
        print(f"Missing keys: {len(missing)}  Unexpected keys: {len(unexpected)}")
    return model

@torch.inference_mode() #redundant with torch.no_grad() but leaving here

def InferenceModel(args,dataloader, model, processor):#, dice_metric
    #training loop
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Available Device: ", device)
    if args.parallel:
        model = DataParallel(model, device_ids=[0,1])
    model.to(device) 
    #create Validation loop
    model.eval()
    dice_list=[]
    batch_num=0
    with torch.no_grad(): # ensure computation graph is created or gradients calculated
        dice = DiceMetric(include_background=False, reduction="none",ignore_empty=False)#mean
        dice_m = DiceMetric(include_background=False, reduction="mean",ignore_empty=False)
        dice_all = DiceMetric(include_background=False, reduction="none",ignore_empty=False)
        binarize = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        imgs_all = []
        for batch in tqdm(dataloader):
            outputs = model(pixel_values = batch["pixel_values"].to(device), \
                            input_boxes = batch["input_boxes"].to(device), \
                            multimask_output = False)
            img_paths = [os.path.splitext(os.path.split(filename)[1])[0] for filename in batch['filename']]
            imgs_all += img_paths
            #get preds and process
            predicted_masks = outputs.pred_masks
            processed_masks = processor.post_process_masks(predicted_masks,batch["original_sizes"],batch["reshaped_input_sizes"],binarize = False)
            masks_post_pred = binarize(processed_masks)
            #get gts
            ground_truth_masks = batch["ground_truth_mask"]
            ground_truth_masks = (ground_truth_masks>0).float().unsqueeze(1).to(device)
            #compute DICE
            dice(y_pred = masks_post_pred,y = ground_truth_masks)
            dice_m(y_pred = masks_post_pred,y = ground_truth_masks)
            dice_all(y_pred = masks_post_pred,y = ground_truth_masks)

            DICE = dice.aggregate()
            mean_DICE = dice_m.aggregate().item()
            dice_list.extend(DICE.detach().cpu().tolist())
            print(f"Batch: {batch_num} Dice mean: {mean_DICE:.3f}")

            B = len(masks_post_pred)
            for idx in range(B):
                pred0 = masks_post_pred[idx].squeeze(0).squeeze(0).float().detach().cpu().numpy()  
                img = batch["raw_image"][idx]            
                gt  = batch["ground_truth_mask"][idx]
                if isinstance(gt, torch.Tensor):
                    gt = gt.detach().cpu()
                if gt.ndim == 3:                         
                    gt = gt.squeeze(0)
                gt = gt.numpy() if hasattr(gt, "numpy") else gt

                # Plot
                fig, axes = plt.subplots(1, 3, figsize=(12, 3), constrained_layout=True)
                axes[0].imshow(img)
                axes[0].set_title(f"Input (batch {batch_num}, idx {idx})")
                axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
                axes[1].set_title("Ground Truth")
                axes[2].imshow(pred0, cmap="gray", vmin=0, vmax=1)
                axes[2].set_title(f"Pred DICE {DICE[idx].item():.3f}")
                for ax in axes: ax.axis("off")
                
                sav_img_base= args.output_save_path 
                save_path = os.path.join(sav_img_base, COMPARISONS_DIR, f"{img_paths[idx]}_Comparison.png")
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # save mask on its own
                np.save(os.path.join(sav_img_base, PREDICTIONS_DIR, f"{img_paths[idx]}_pred") , pred0)
            batch_num+=1
            dice.reset(); dice_m.reset()
        dice_all = np.array(dice_list).flatten()
        res = bootstrap((dice_all,),statistic=np.mean,n_resamples=1000,confidence_level=0.95,method="percentile",random_state=0)
        mean = float(dice_all.mean())
        lo = float(res.confidence_interval.low)
        hi = float(res.confidence_interval.high)
        print(f"FINAL DICE: mean: {mean:.3f} 95%CI [{lo:.3f}:{hi:.3f}]")

        pd.DataFrame([{'img': img, 'dice': dice} for img, dice in zip(imgs_all, dice_all)]).to_csv(os.path.join(sav_img_base, 'dice_scores.csv'), index=False)

def parse_args():
    parser = argparse.ArgumentParser()
    #Path to inference csv
    parser.add_argument("--inference_data_path", type=str, required=True,
                        help="Path to inference dataset (CSV)")
    #image path and GT mask path
    parser.add_argument("--image_col", type=str, default="image_path",
                        help="Column name for image files in the train data")
    parser.add_argument("--mask_col", type=str, default="mask_path",
                        help="Column name for mask files in the train data")
    parser.add_argument("--base_model", type=str, default="wanglab/medsam-vit-base",
                        help="Medsam basemodel for training wanglab default. Also flaviagiammarino/medsam-vit-base")
    parser.add_argument("--model_ckpt", type=str, default="ckpt.pth",
                        help="Path to model weights")
    parser.add_argument("--image_size", type=int, nargs=2, metavar=("W","H"), default=(1024,1024),
                        help="SAM Model resizes images to 1024 so load at 1024")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Batch size (10max for l40)")
    parser.add_argument("--parallel", type=bool, default=True,
                        help="Default True 2 GPU DataParallel, False 1 GPU.")
    parser.add_argument("--output_save_path", type=str, default="./checkpoints",
                        help="Directory to save inference")
    return parser.parse_args()

def main():
    args = parse_args()
    print("Inference dataset:", args.inference_data_path)
    print("Image size:", args.image_size)
    print("Batch size:", args.batch_size)
    print("Saving output to:", args.output_save_path)
    print("Loading Base Model:", args.base_model)
    #import model and model weights
    model = SamModel.from_pretrained(args.base_model)
    proc = SamProcessor.from_pretrained(args.base_model)
    model = load_any_ckpt(model,args.model_ckpt)
    #import data
    inf_df = pd.read_csv(args.inference_data_path)
    img_paths = inf_df[args.image_col]
    gt_paths = inf_df[args.mask_col]

    os.makedirs(os.path.join(args.output_save_path, COMPARISONS_DIR), exist_ok=True)
    os.makedirs(os.path.join(args.output_save_path, PREDICTIONS_DIR), exist_ok=True)

    print("Loading Data...")
    dataloader = LoadData(img_paths,gt_paths,proc,image_size=args.image_size,batch_size=args.batch_size)
    print("Data Loaded")
    print("Running Inference...")
    InferenceModel(args,dataloader, model, proc)

if __name__ == "__main__":
    main()