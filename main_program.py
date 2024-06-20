import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# PyTorch Libraries
import torch
from tqdm import tqdm
from IPython.display import clear_output
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# Back Bone
import segmentation_models_pytorch as smp

# Local
from json_handler import JSON_handler
from loss_function import DiceLoss
from model import ModifiedUnet

# Define Constants
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8
NUM_EPOCHS = 35
LEARNING_RATE =1e-3

# Define Classes
            
class CustomDataset_general(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_folder = os.path.join(root_dir, "images")
        self.mask_folder = os.path.join(root_dir, "masks")
        self.image_files = sorted(os.listdir(self.image_folder))
        self.mask_files = sorted(os.listdir(self.mask_folder))
        self.transform = transform

    def __len__(self):
        return len(self.mask_files)

    def __getitem__(self, idx):
        # Read image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert("RGB")
        image_gray = image.convert("L")  # Convert to grayscale

        # Read corresponding mask
        mask_name = self.mask_files[idx]
        mask_path = os.path.join(self.mask_folder, mask_name)
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            # Apply transformations
            image_gray = self.transform(image_gray)
            mask = self.transform(mask)

        return image_gray, mask 

def create_mask() :
    # Define the paths
    original_image_dir = 'Image/Brain_Tumor_Dataset/train'  
    json_file = 'Image/Brain_Tumor_Dataset/train/_annotations.coco.json'
    mask_output_folder = 'Image/processed/train/masks' 
    image_output_folder = 'Image/processed/train/images' 

    print("Creating train masks...")
    train_json_handler = JSON_handler(original_image_dir, json_file, image_output_folder, mask_output_folder)
    train_json_handler.json_to_mask()
    
    original_image_dir = 'Image/Brain_Tumor_Dataset/valid'
    json_file = 'Image/Brain_Tumor_Dataset/valid/_annotations.coco.json'
    mask_output_folder = 'Image/processed/valid/masks'
    image_output_folder = 'Image/processed/valid/images'
    
    print("Creating validation masks...")
    valid_json_handler = JSON_handler(original_image_dir, json_file, image_output_folder, mask_output_folder)
    valid_json_handler.json_to_mask()
    
    original_image_dir = 'Image/Brain_Tumor_Dataset/test'
    json_file = 'Image/Brain_Tumor_Dataset/test/_annotations.coco.json'
    mask_output_folder = 'Image/processed/test/masks'
    image_output_folder = 'Image/processed/test/images'
    
    print("Creating test masks...")
    test_json_handler = JSON_handler(original_image_dir, json_file, image_output_folder, mask_output_folder)
    test_json_handler.json_to_mask()
    
    return

def create_loader() :
    train_path = "Image/processed/train" 
    valid_path = "Image/processed/valid"
    test_path = "Image/processed/test"

    image_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),  # Assuming grayscale images
        transforms.Lambda(lambda x: x.clamp(0, 1))
    ])

    print("Creating datasets for loaders...")
    # Create datasets
    train_dataset = CustomDataset_general(train_path, transform=image_transform)
    valid_dataset = CustomDataset_general(valid_path, transform=image_transform)
    test_dataset = CustomDataset_general(test_path, transform=image_transform)

    print("Creating all data loaders...")
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, valid_loader, test_loader

def create_model(model_backbone, unfreeze = False, enc_name = 'efficientnet-b0', encoder_weights = 'imagenet', in_channels = 1, classes = 1, activation = None) :
    model = model_backbone(
        encoder_name=enc_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
    )
    
    if unfreeze:
        for param in model.encoder.parameters():
            param.requires_grad = True
    return model

def user_select_model() :
    print("Select Model")
    print("1. ModifiedUnet UnFrozen")
    print("2. ModifiedUnet Frozen Encoder")
    print("3. EfficientUNet++")
    print("Select the model you want to use : ", end="")
    selection = int(input())
    
    if selection == 1 :
         model = create_model(ModifiedUnet)
    elif selection == 2 :
        model = create_model(ModifiedUnet, unfreeze = True)
    elif selection == 3 :
        model = create_model(smp.EfficientUnetPlusPlus, enc_name='timm-efficientnet-b5')
    else :
        print("Invalid Selection")
        return None
        
    return model

def compute_iou(outputs, targets):
    intersection = torch.logical_and(outputs, targets).sum()
    union = torch.logical_or(outputs, targets).sum()
    iou = intersection.float() / union.float()
    return iou.item()

def train(model, optimizer, loss_fn, loader, device):
    epoch_loss = 0.0
    epoch_iou = 0.0
    model.train()

    for x, y in tqdm(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        
        masks = model(x)  # Split the output of the model
        loss = loss_fn(masks, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        preds = (masks > 0.8).float() 
        iou = compute_iou(preds, y)
        epoch_iou += iou

    return epoch_loss / len(loader), epoch_iou / len(loader)

def valid(model, loader, loss_fn, device):
    epoch_loss = 0.0
    epoch_iou = 0.0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            
            masks = model(x) 
            loss = loss_fn(masks, y)
            epoch_loss += loss.item()
            
            # Compute IoU
            preds = (masks > 0.8).float()  
            iou = compute_iou(preds, y)
            epoch_iou += iou

    return epoch_loss / len(loader), epoch_iou / len(loader)

def model_training(optimizer, model, train_loader, valid_loader, loss_fn, checkpoints_path): 
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []
    num_no_improve = 0

    model.to(DEVICE)  

    for epoch in range(NUM_EPOCHS):
        train_loss, train_iou = train(model, optimizer, loss_fn, train_loader, DEVICE) 
        val_loss, val_iou = valid(model, valid_loader, loss_fn, DEVICE) 
        
        train_losses.append(train_loss) 
        val_losses.append(val_loss)  
        train_ious.append(train_iou)  
        val_ious.append(val_iou)  
        
        clear_output(wait=True)
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Validation IoU: {val_iou:.4f}')
        
        if val_loss < best_val_loss:  
            best_val_loss = val_loss
            print("Saving the model")
            torch.save(model.state_dict(), checkpoints_path) 
        
        if val_loss > best_val_loss:
            best_dict = torch.load(checkpoints_path)
            print('Loading Best Model')
            model.load_state_dict(best_dict)
        
        if len(val_losses) > 1: 
            if val_loss <= best_val_loss: 
                num_no_improve += 1
            else:
                num_no_improve = 0
                
        if num_no_improve >= 10:
            print("Early stopping")
            break
        
    return train_losses, val_losses, train_ious, val_ious

def plot_train_stat(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training Loss and Validation Loss')
    plt.legend()
    plt.show(block=False)

def main() :
    model = None
    print("Step 1 : Creating masks...")
    create_mask()
    
    print("Step 2 : Creating loaders...")
    train_loader, valid_loader, test_loader = create_loader()
    
    while model is None  :
        model = user_select_model()
    model.to(DEVICE)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    print("Step 3 : Training the model...")
    # Create Checkpoint
    checkpoints_path = "exp/checkpoints.pth"
    os.makedirs("exp", exist_ok=True)  
    loss_fn = DiceLoss()
    
    train_losses, val_losses, train_ious, val_ious = model_training(optimizer, model, train_loader, valid_loader, loss_fn, checkpoints_path)
    
    print("Step 4 : Plotting the training statistics...")
    plot_train_stat(train_losses, val_losses)
    
    print("Step 5 : Testing the model...")
    model.load_state_dict(torch.load(checkpoints_path))
    model.to(DEVICE)
    
    test_loss, test_iou = valid(model, test_loader, loss_fn, DEVICE)
    print(f'Test Loss: {test_loss:.4f}, Test IoU: {test_iou:.4f}')
    
    return 

main()
    


    
        
    
            
            
    

    
    
