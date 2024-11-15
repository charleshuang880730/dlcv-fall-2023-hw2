# -*- coding: utf-8 -*-
# Reference: https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php Hw3 sample code
_exp_name = "Part3_SOURCEONLY_LOW"

# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import itertools

# This is for the progress bar.
from tqdm.auto import tqdm
import random
# import wandb
from part3_model import DANN

myseed = 777  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

# wandb.init(project="DLCV_Hw1P1", entity="charles0730")


"""# Datasets
The data is labelled by the name, so we load images and label while calling '__getitem__'
"""

class Dataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_folder,
        image_size
    ):
        # Read the csv file
        self.data_info = pd.read_csv(csv_file)
        
        # Store the image names and labels
        self.image_names = self.data_info['image_name'].tolist()
        self.labels = self.data_info['label'].tolist()
        
        self.img_folder = img_folder
        self.image_size = image_size
        
        # Data augmentation
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        # Get image name and label from the pandas df
        single_image_name = self.image_names[index]
        single_image_label = self.labels[index]
        
        # Open image and apply transformations
        img_path = Path(self.img_folder) / single_image_name
        img = Image.open(img_path).convert("RGB")
        
        # Transform image
        img = self.transform(img)
        
        return img, single_image_label
        

"""# Configurations"""

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"
# Initialize a model, and put it on the device specified.
model = DANN().to(device)
# The number of batch size.
batch_size = 512
# The number of training epochs.
n_epochs = 100
# If no improvement in 'patience' epochs, early stop.
patience = 20
# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()
# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
# Learning rate scheduler
# scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-2)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# wandb.watch(model, log="all")

"""# Dataloader"""

# Construct train and valid datasets.
source_set = Dataset(csv_file="../hw2_data/digits/svhn/train.csv", img_folder="../hw2_data/digits/svhn/data", image_size=(28, 28))
source_loader = DataLoader(source_set, batch_size=batch_size, shuffle=True, num_workers=5)
# Validation set from target domain
target_valid_set = Dataset(csv_file="../hw2_data/digits/svhn/val.csv", img_folder="../hw2_data/digits/svhn/data", image_size=(28, 28))
target_valid_loader = DataLoader(target_valid_set, batch_size=batch_size, shuffle=False, num_workers=5)

"""# Start Training"""

# Initialize trackers, these are not parameters and should not be changed
stale = 0
best_acc = 0

for epoch in range(n_epochs):

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    for source_data, source_label in tqdm(source_loader):
        
        source_data, source_label = source_data.to(device), source_label.to(device)

        logits = model(source_data)

        loss = criterion(logits, source_label)

        # Zero the optimizer gradients
        optimizer.zero_grad()

        # Combine losses and backpropagate
        loss.backward()
        optimizer.step()

        acc = (logits.argmax(dim=-1) == source_label).float().mean()

        # Log training metrics
        train_loss.append(loss.item())
        train_accs.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for batch in tqdm(target_valid_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        #imgs = imgs.half()

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            # logits = model(imgs.to(device))
            logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)
        #break

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

    # wandb.log({"Train Loss": train_loss, "Train Accuracy": train_acc})
    # wandb.log({"Validation Loss": valid_loss, "Validation Accuracy": valid_acc})

    # update logs
    if valid_acc > best_acc:
        with open(f"./{_exp_name}_log.txt","a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
    else:
        with open(f"./{_exp_name}_log.txt","a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


    # save models
    if valid_acc > best_acc:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), f"{_exp_name}_best.ckpt") # only save best to prevent output memory exceed error
        best_acc = valid_acc
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvment {patience} consecutive epochs, early stopping")
            break

    scheduler.step()

# wandb.finish()