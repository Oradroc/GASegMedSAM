import argparse
import os
import cv2
import numpy as np
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
from transformers import SamModel, SamProcessor

PREDICTIONS_DIR = 'predictions'

def entire_image_bounding_box(size):
    W, H = size
    bbox = [0, 0, W, H]#np.array([0, 0, W, H])
    return bbox
def to_rgb(img):
    '''
    Likely this channel transformation is unecessary, 
    but I want to ensure everything is RGB for SAM which assumes RGB
    '''
    if img is None:
        return None
    if img.ndim == 2: #grayscale to RGB
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#BGR -> RGB
class MedSAMDataset(TorchDataset):
    def __init__(self,args, dataset, processor, image_size):
        self.dataset = dataset
        self.processor = processor
        self.size = image_size
        self.args=args
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        img_raw = cv2.imread(item["image"], cv2.IMREAD_UNCHANGED)
        img_raw = to_rgb(img_raw)
        img_raw = cv2.resize(img_raw, self.size, interpolation=cv2.INTER_LINEAR)

        image = np.array(img_raw, dtype=np.uint8)

        prompt = entire_image_bounding_box(self.size)
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["raw_image"] = image
        inputs["filename"] = item["image"]

        return inputs

def LoadData(args, inf_df, processor, image_size=(1024, 1024), batch_size=10):
    '''
    With imread unchanged reading in images with 4 channels as a BGR
    need 3 channel RGB for sam
    For masks need to use inter_nearest to ensure 0/1 does not get blended into gray
    '''
    datadict = {"image": inf_df[args.image_col]}
    data = HFDataset.from_dict(datadict)
    dataset = MedSAMDataset(args=args, dataset=data, processor=processor, image_size=image_size)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False,pin_memory=True)
    return dataloader

def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            print("BrokenStateDict")  
    else:
        state = ckpt #already a raw state_dict

    #strip common wrappers
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
        ckpt = torch.load(path, map_location=device)#PyTorch 2.6 defaults to weights_only=True
    except Exception:
        ckpt = torch.load(path, map_location=device, weights_only=False)#trusted only

    state = _extract_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing or unexpected:
        print(f"Missing keys: {len(missing)}  Unexpected keys: {len(unexpected)}")
    return model

@torch.inference_mode()#redundant with torch.no_grad() but leaving here
def InferenceModel(args, dataloader, model, processor):
    #Inference loop
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Available Device: ", device)
    if args.parallel and torch.cuda.is_available() and torch.cuda.device_count()>1:
        model = DataParallel(model)
    model.to(device)
    #create inference loop
    model.eval()
    with torch.no_grad():#disable gradient tracking
        for batch_num, batch in enumerate(tqdm(dataloader)):
            #get inputs
            pixel_values = batch["pixel_values"].to(device)
            input_boxes = batch["input_boxes"].to(device)
            outputs = model(pixel_values=pixel_values,input_boxes=input_boxes,multimask_output=False)
            img_paths = [os.path.splitext(os.path.split(filename)[1])[0]for filename in batch["filename"]]
            #get preds and process
            predicted_masks = outputs.pred_masks
            processed_masks = processor.post_process_masks(predicted_masks,batch["original_sizes"],batch["reshaped_input_sizes"],binarize=False)
            #save mask
            for idx, pred in enumerate(processed_masks):
                pred0 = torch.sigmoid(pred)
                pred0 = (pred0>0.5).squeeze().detach().cpu().numpy().astype(np.uint8)
                np.save(os.path.join(args.output_save_path, PREDICTIONS_DIR, f"{img_paths[idx]}_pred.npy"),pred0)

            print(f"Finished batch {batch_num}")

def parse_args():
    parser = argparse.ArgumentParser()
    #Path to inference csv
    parser.add_argument("--inference_data_path", type=str, required=True,
                        help="Path to inference dataset (CSV)")
    #image path
    parser.add_argument("--image_col", type=str, default="image_path",
                        help="Column name for image files in the inference data")
    #base model and weights
    parser.add_argument("--base_model", type=str, default="wanglab/medsam-vit-base",
                        help="Medsam basemodel for inference. Also flaviagiammarino/medsam-vit-base")
    parser.add_argument("--model_ckpt", type=str, default="ckpt.pth",
                        help="Path to model weights")
    #image specs and batch size
    parser.add_argument("--image_size", type=int, nargs=2, metavar=("W","H"), default=(1024,1024),
                        help="SAM Model resizes images to 1024 so load at 1024")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Batch size (10max for l40)")
    parser.set_defaults(parallel=True)
    parser.add_argument("--parallel", dest="parallel", action="store_true",
                        help="Use DataParallel when multiple GPUs are available")
    parser.add_argument("--no_parallel", dest="parallel", action="store_false",
                        help="Disable DataParallel and use a single GPU")
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
    model = load_any_ckpt(model, args.model_ckpt)
    #import data
    inf_df = pd.read_csv(args.inference_data_path)
    os.makedirs(os.path.join(args.output_save_path, PREDICTIONS_DIR), exist_ok=True)
    print("Loading Data...")
    dataloader = LoadData(args, inf_df, proc, image_size=args.image_size, batch_size=args.batch_size)
    print("Data Loaded")
    print("Running Inference...")
    InferenceModel(args, dataloader, model, proc)

if __name__ == "__main__":
    main()