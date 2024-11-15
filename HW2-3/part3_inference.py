# -*- coding: utf-8 -*-

# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import argparse
import os

# This is for the progress bar.
from tqdm.auto import tqdm
from part3_model import DANN

myseed = 777  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

"""# Datasets
The data is labelled by the name, so we load images and label while calling '__getitem__'
"""

class Dataset(Dataset):
    def __init__(self, img_folder, image_size):
        # Initialize the attributes
        self.img_folder = Path(img_folder)
        self.image_size = image_size
        self.image_names = sorted([img_name for img_name in os.listdir(img_folder) if img_name.endswith(('png', 'jpg', 'jpeg'))])
        
        # Data augmentation
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        # Get image name
        single_image_name = self.image_names[index]
        
        # Open image and apply transformations
        img_path = self.img_folder / single_image_name
        img = Image.open(img_path).convert("RGB")
        
        # Transform image
        img = self.transform(img)
        
        return img, single_image_name
        

"""# Configurations"""

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"
# Initialize a model, and put it on the device specified.
model = DANN().to(device)
# The number of batch size.
batch_size = 512

argparser = argparse.ArgumentParser()
argparser.add_argument("--input", type=str, help="Input file path.")
argparser.add_argument("--output", type=str, default="output.csv", help="Output file path.")
args = argparser.parse_args()

# wandb.watch(model, log="all")

"""# Dataloader"""

# Validation set from target domain
target_test_set = Dataset(img_folder=f"{args.input}", image_size=(28, 28))
target_test_loader = DataLoader(target_test_set, batch_size=batch_size, shuffle=False, num_workers=6)

""" Testing and generate prediction CSV """

model_best = DANN().to(device)

if 'svhn' in f"{args.input}":
    model_best.load_state_dict(torch.load("./Part3_DANN_SVHN_best.ckpt", map_location=device))
elif 'usps' in f"{args.input}":
    model_best.load_state_dict(torch.load("./Part3_DANN_USPS_best.ckpt", map_location=device))
model_best.eval()

image_filenames = target_test_set.image_names

predictions = []
image_filenames = []
with torch.no_grad():
    for images, image_names in tqdm(target_test_loader):
        test_pred = model_best(images.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        predictions.extend(test_label.tolist())
        image_filenames.extend(image_names)

df = pd.DataFrame({
    'image_name': image_filenames,
    'label': predictions
})

df.to_csv(f"{args.output}", index=False)