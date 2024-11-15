# -*- coding: utf-8 -*-

# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from sklearn.manifold import TSNE
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import argparse

from part3_model import DANN


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
model.load_state_dict(torch.load("Part3_DANN_USPS_best.ckpt"))
model.eval()
# The number of batch size.
batch_size = 512

"""# Dataloader"""

# Construct train and valid datasets.
source_set = Dataset(csv_file="../hw2_data/digits/mnistm/val.csv", img_folder="../hw2_data/digits/mnistm/data", image_size=(28, 28))
source_loader = DataLoader(source_set, batch_size=batch_size, shuffle=True, num_workers=6, drop_last=True)
target_set = Dataset(csv_file="../hw2_data/digits/usps/val.csv", img_folder="../hw2_data/digits/usps/data", image_size=(28, 28))
target_loader = DataLoader(target_set, batch_size=batch_size, shuffle=True, num_workers=6, drop_last=True)

# Extract features and labels
features_list = []
digit_labels_list = []
domain_labels_list = []

for data in source_loader:
    inputs, digit_labels = data
    inputs = inputs.to(device)
    with torch.no_grad():
        features = model.backbone(inputs).cpu().numpy()
        features_list.append(features)
        digit_labels_list.append(digit_labels.numpy())
        domain_labels_list.append(np.zeros((len(inputs),), dtype=np.int32))

for data in target_loader:
    inputs, digit_labels = data
    inputs = inputs.to(device)
    with torch.no_grad():
        features = model.backbone(inputs).cpu().numpy()
        features_list.append(features)
        digit_labels_list.append(digit_labels.numpy())
        domain_labels_list.append(np.ones((len(inputs),), dtype=np.int32))

# Concatenate all features and labels
features = np.concatenate(features_list, axis=0)
features = features.reshape(features.shape[0], -1)  # Flatten the features
digit_labels = np.concatenate(digit_labels_list, axis=0)
domain_labels = np.concatenate(domain_labels_list, axis=0)

fig, axes = plt.subplots(1, 2, figsize=(20, 9))
all_feature = TSNE(2, init='pca', learning_rate='auto').fit_transform(features)
scatter = axes[0].scatter(
    all_feature[..., 0], all_feature[..., 1],
    c=digit_labels, alpha=0.5, s=10
)
axes[0].legend(*scatter.legend_elements(), title='Digits')
axes[0].set_title("Colored by Different Classes")

scatter = axes[1].scatter(
    all_feature[..., 0], all_feature[..., 1],
    c=domain_labels, alpha=0.5, s=10
)
axes[1].legend(handles=scatter.legend_elements()[0], labels=[
    'Source', 'Target'], title='Domains')
axes[1].set_title("Colored by Different Domains")
fig.savefig("USPS")

"""# Run t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(features)

# Plotting function
def plot_with_labels(low_dim_embs, labels, title, filename=None):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(10, 10))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(str(label),
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.title(title)
    if filename:
        plt.savefig(filename)

# Plot by digit class
plot_with_labels(tsne_results, digit_labels, 't-SNE colored by digit class')

# Plot by domain
plot_with_labels(tsne_results, domain_labels, 't-SNE colored by domain')"""