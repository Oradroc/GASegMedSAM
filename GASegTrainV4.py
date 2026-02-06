
#Patch: transformers SAM dropout property (import bug workaround)
from transformers.models.sam.modeling_sam import SamAttention
import torch
import torch.nn as nn
def _get_dropout_p(self):
    for m in self.modules():
        if isinstance(m, nn.Dropout):
            return m.p
    return 0.0
SamAttention.dropout_p = property(_get_dropout_p)
import argparse
import os
import random
import sys
from collections import OrderedDict
from itertools import chain
from statistics import mean
import numpy as np
import pandas as pd
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.nn import DataParallel
from torch.nn.functional import interpolate, normalize, threshold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.utils.tensorboard import SummaryWriter
import monai
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets import Dataset as HFDataset
from transformers import SamModel, SamProcessor


def entire_image_bounding_box(image,size):
    H, W = size
    bbox = [0, 0, W, H]#np.array([0, 0, W, H])
    return bbox

def getTransform():
    '''
    Can modify. Was not noticing real performance improvements with transformation param tuning.
    Included translations to help with edge segmentations. 
    Some of these transformations are less real world since fundus images typically have the same orientation
    Applied anyways to make model more robust
    '''

    RandTransform = A.Compose([
                            A.Affine(
                                translate_percent={"x": (-0.25, 0.25), "y": (-0.25, 0.25)},  # up to ±25% width/height
                                shear={"x": (-10, 10), "y": (-10, 10)},                      # 10 degrees of shear
                                rotate=0.0,
                                scale=1.0,
                                fit_output=False,
                                p=0.5
                            ),
                            A.OneOf([
                            A.Rotate(limit=30,p=0.5),
                            A.RandomRotate90(p=0.5),
                            A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.5)],
                            p=0.9)])
    return RandTransform

def to_rgb(img):
    '''
    Likely this channel transformation is unecessary, 
    but I want to ensure everything is RGB for SAM which assumes GGB
    '''
    if img is None:
        return None
    if img.ndim == 2: # grayscale to RGB (in case img is grayscale convert to RGB for SAM)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB

class MedSAMDataset(TorchDataset):
    def __init__(self, args, dataset, processor):
        self.dataset = dataset
        self.processor = processor
        self.transform = args.transforms
        self.size = args.image_size
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx):
        item = self.dataset[idx]
        #read img on and resize
        img_raw = cv2.imread(item["image"],cv2.IMREAD_UNCHANGED)
        img_raw = to_rgb(img_raw) #convert cv2 BGR channels to RGB which is SAM input, likely would not alter performance
        img_raw = cv2.resize(img_raw, self.size, interpolation=cv2.INTER_LINEAR) #use inter_linear for images bc continuous
        #read mask on and resize
        mask_raw = cv2.imread(item["label"],cv2.IMREAD_GRAYSCALE)
        mask_raw = cv2.resize(mask_raw, self.size, interpolation=cv2.INTER_NEAREST) #must use inter_nearest for binary masks to preserve exact pixel identity
        #make image and gt 8bit numpy arrays 
        image = np.array(img_raw, dtype=np.uint8)
        ground_truth_mask = np.array(mask_raw,dtype=np.uint8)
        #Add image transforms if hyperparam is True
        if self.transform:
            T = getTransform()
            #apply same transformation function to both image and mask
            augmented = T(image=image, mask=ground_truth_mask)
            image = augmented["image"]
            ground_truth_mask = augmented["mask"]
        #obtain bounding box for model prompt (bbox is image size not)
        prompt = entire_image_bounding_box(ground_truth_mask,self.size)#get size of mask which is same size as image
        #prompt = get_bounding_box(ground_truth_mask) # Puts one tight bounding box around mask (cheating unless you have a YOLO detection to get input) entire image modeled converged to tight bbox model when experimenting, so unecessary.
        #prepare for model
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors = "pt") #input boxes are list of lists bc you can have multiple prompts and input labels pos and neg point prompts. Default pos
        #remove dimensions of the batch
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        #add ground truth 
        inputs["ground_truth_mask"] = ground_truth_mask
        return inputs

#chatgpt written strip function
def _strip_module_prefix(state_dict):
    # Handle checkpoints saved under DataParallel/DDP ("module." prefix)
    if state_dict and next(iter(state_dict)).startswith("module."):
        return OrderedDict((k.replace("module.", "", 1), v) for k, v in state_dict.items())
    return state_dict

def resume_epoch(args,model,optimizer,device):
    '''
    Load prior checkpoint and load model
    return the new epoch to start on
    '''
    ckpt = torch.load(args.path_to_chkpt, map_location=device)
    state = ckpt["model_state_dict"]
    state = _strip_module_prefix(state)
    #load model
    model.load_state_dict(state, strict=False)
    model.to(device)
    #load optimizer
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    #get_epoch
    epoch = ckpt["epoch"]
    start_epoch = epoch+1
    return start_epoch

def LoadData(args,img_paths,gt_paths,processor):
    '''
    With imread unchanged reading in images with 4 channels as a BGR
    need 3 channel RGB for sam
    For masks need to use inter_nearest to ensure 0/1 does not get blended into gray
    '''
    datadict = {"image": img_paths,"label": gt_paths}
    data = HFDataset.from_dict(datadict)
    dataset = MedSAMDataset(args, dataset=data, processor=processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,shuffle=True)
    return dataloader

def TrainModel(args,training_dataloader, validation_dataloader, model, processor, optimizer, seg_loss):#, dice_metric
    #training loop
    num_epochs = args.num_epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Available Device: ", device)
    #Check if resuming prior epoch
    start_epoch=0
    if args.resume_chkpt:
        start_epoch = resume_epoch(args,model,optimizer,device)
    
    #Not true parallel. Does not scale well past 2 GPUs.
    if args.parallel:
        model = DataParallel(model, device_ids=[0,1])
    model.to(device)
    model.train()
    #initialize tensorboard writer
    writer = SummaryWriter(log_dir=args.tensorboard_log_dir)

    for epoch in range(start_epoch,num_epochs):
        torch.cuda.empty_cache()
        model.train() #restart training after validation
        train_losses=[]
        for batch in tqdm(training_dataloader):
            optimizer.zero_grad() #keep gradients from prior iteration from accumulating
            #fwd pass and create computation graph 
            outputs = model(pixel_values = batch["pixel_values"].to(device), \
                            input_boxes = batch["input_boxes"].to(device), \
                            multimask_output = False)
            #compute loss 
            predicted_masks = outputs.pred_masks.squeeze(1) #squeeze batch channel
            #SAM model inference output is 256,256 so need to resize to 1024,1024 for gt comparison
            predicted_masks = interpolate(
                predicted_masks, 
                size=args.image_size,      
                mode="bilinear", 
                align_corners=False
            )#using bilinear bc outputs are logits
            #get original ground truths and send to device
            ground_truth_masks = batch["ground_truth_mask"]
            ground_truth_masks = (ground_truth_masks>0).float().unsqueeze(1).to(device)
            #calculate loss using monai dice loss specified at initialization
            loss = seg_loss(predicted_masks,ground_truth_masks)
            #backward pass to compute derivatives of loss to get gradients
            loss.backward()
            #Update params based on gradients
            optimizer.step()
            #track loss
            train_losses.append(loss.item())
        
        #Validation loop
        model.eval() #switch model to eval mode
        val_losses = []
        with torch.no_grad(): # ensure no computation graph is created or gradients calculated
            for batch in tqdm(validation_dataloader):
                outputs = model(pixel_values = batch["pixel_values"].to(device), \
                                input_boxes = batch["input_boxes"].to(device), \
                                multimask_output = False)
                #compute loss
                predicted_masks = outputs.pred_masks.squeeze(1)
                #SAM model inference output is 256,256 so need to resize to 1024,1024 for gt comparison
                predicted_masks = interpolate(
                predicted_masks, 
                size=args.image_size,      
                mode="bilinear", 
                align_corners=False
                )#using bilinear bc outputs are logits
                ground_truth_masks = batch["ground_truth_mask"]
                ground_truth_masks = (ground_truth_masks>0).float().unsqueeze(1).to(device)

                loss = seg_loss(predicted_masks,ground_truth_masks)
                val_losses.append(loss.item())

        #create checkpoint
        train_losses = mean(train_losses)
        val_losses = mean(val_losses)
       
        checkpoint = {
            "epoch":epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_losses,
            "val_loss": val_losses,
            "val_dice_metric": 1-val_losses #approximate
        }
        
        #write to torch
        writer.add_scalar("Loss/Train", train_losses, epoch)
        writer.add_scalar("Loss/Validation", val_losses, epoch)
        writer.add_scalar("SoftDICE/Validation", (1-val_losses), epoch)
        ckpt_path = os.path.join(args.model_save_path, f"ckpt-epoch{epoch:03d}.pth")
        torch.save(checkpoint, ckpt_path)

    writer.close()

def TuneMedSAM(args):
    print("Initializing model and Loading Data:")
    #Load base medsam processor and model
    processor = SamProcessor.from_pretrained(args.base_model)# wanglab/medsam-vit-base other flaviagiammarino/medsam-vit-base
    model = SamModel.from_pretrained(args.base_model)
    #Load training data
    train_df = pd.read_csv(args.train_data_path)
    train_img_paths = train_df[args.train_image_col]
    train_gt_paths = train_df[args.train_mask_col]
    training_dataloader = LoadData(args,train_img_paths,train_gt_paths,processor)
    print("Training Data Loaded")
    #Load val data
    val_df = pd.read_csv(args.val_data_path)
    val_img_paths = val_df[args.val_image_col]
    val_gt_paths = val_df[args.val_mask_col]
    validation_dataloader = LoadData(args,val_img_paths,val_gt_paths,processor)
    print("Validation Data Loaded")
    #initialize optimizer and loss function
    #do not update prompt encoder, only image (vision) and mask_decoder
    #freeze prompt encoder
    #explicitly exclude prompt encoder
    for p in model.prompt_encoder.parameters():
        p.requires_grad = False
    #explicitly include vision_encoder and mask_decoder
    #commented out code is in case you want to tune the hyperparams individually at different lr (not currently set up for this, would need lr arg for each param in args)
    '''optimizer = AdamW([{"params":(p for p in sam.vision_encoder.parameters() if p.requires_grad),"lr":args.lr},
     {"params": (p for p in sam.mask_decoder.parameters() if p.requires_grad), "lr": args.lr}],
     weight_decay=args.weight_decay)'''
    params = chain(model.vision_encoder.parameters(), model.mask_decoder.parameters())
    optimizer = AdamW(params,lr=args.lr,weight_decay=args.weight_decay)
    #Use Monai Dice loss function
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    #Check cuda memory allocation to make sure it is clear (convert bytes to mebibytes)
    print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MiB")
    print("Reserved:",  torch.cuda.memory_reserved()  / 1024**2, "MiB")
    print("Training Model...")
    TrainModel(args,training_dataloader, validation_dataloader, model, processor, optimizer, seg_loss)

def parse_args():
    parser = argparse.ArgumentParser()
    #Paths to taining and val data and column labels
    #Train and Val data should be two separate CSVs
    parser.add_argument("--train_data_path", type=str, required=True,
                        help="Path to training dataset (CSV)")
    parser.add_argument("--val_data_path", type=str, required=True,
                        help="Path to validation dataset (CSV)")
    #Two separate inputs for train and val file img/mask labels in case they are different (img_train/img_val)
    parser.add_argument("--train_image_col", type=str, default="image_path",
                        help="Column name for image files in the train data")
    parser.add_argument("--train_mask_col", type=str, default="mask_path",
                        help="Column name for mask files in the train data")
    parser.add_argument("--val_image_col", type=str, default="image_path",
                        help="Column name for image files in the val data")
    parser.add_argument("--val_mask_col", type=str, default="mask_path",
                        help="Column name for mask files in the val data")

    # Model saving and logging to tensorboard
    parser.add_argument("--model_save_path", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--tensorboard_log_dir", type=str, default="./runs",
                        help="Directory for TensorBoard logs")

    # tuning and hyper-parameters
    parser.add_argument("--base_model", type=str, default="wanglab/medsam-vit-base",
                        help="Medsam basemodel for training wanglab default. Also flaviagiammarino/medsam-vit-base")
    parser.add_argument("--resume_chkpt",type=bool,default=False,
                        help="Flag for resuming from checkpoint")
    parser.add_argument("--path_to_chkpt",type=str,default=None,
                        help="Path to the check point restarting at")
    parser.add_argument("--parallel", type=bool, default=True,
                        help="Default True 2 GPU DataParallel, False 1 GPU.")
    parser.add_argument("--image_size", type=int, nargs=2, metavar=("W","H"), default=(1024,1024),
                        help="SAM Model resizes images to 1024 so load at 1024")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Batch size (10max for l40)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="Weight decay for optimizer")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Total number of training epochs")
    parser.add_argument("--transforms", type=bool, default=True,
                        help="Apply random transformations to training images")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.model_save_path, exist_ok=True) #False to ensure prior model is not accidentally overwritten
    os.makedirs(args.tensorboard_log_dir, exist_ok=True)
    print("Training dataset:", args.train_data_path)
    print("Validation dataset:", args.val_data_path)
    print("Image size:", args.image_size)
    print("Batch size:", args.batch_size)
    print("LR:", args.lr)
    print("Weight Decay:", args.weight_decay)
    print("Epochs:", args.num_epochs)
    print("Apply transforms:", args.transforms)
    print("Saving checkpoints to:", args.model_save_path)
    print("TensorBoard logs in:", args.tensorboard_log_dir)
    TuneMedSAM(args)

if __name__ == "__main__":
    main()