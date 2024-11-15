from pathlib import Path
from torchvision import transforms as T, utils

from PIL import Image
from tqdm.auto import tqdm
# from ema_pytorch import EMA

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image, make_grid
import pandas as pd
from Conditional_UNet import ContextUnet
from ddpm import DDPM

torch.backends.cudnn.benchmark = True
torch.manual_seed(4096)

if torch.cuda.is_available():
  torch.cuda.manual_seed(4096)

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

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
        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        # Get image name and label from the pandas df
        single_image_name = self.image_names[index]
        single_image_label = self.labels[index]
        
        # Open image and apply transformations
        img_path = Path(self.img_folder) / single_image_name
        img = Image.open(img_path)
        
        # Transform image
        img = self.transform(img)
        
        return img, single_image_label

# ---------- Training ----------
n_epoch = 50
batch_size = 512
n_T = 400
device = "cuda:0"
n_classes = 10
n_feat = 128 # 128 ok, 256 better (but slower)
lrate = 1e-4
save_model = True
save_dir = './part1_train_output/'
w = 2.0 # strength of generative guidance

ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
ddpm.to(device)

dataset = Dataset(csv_file="./hw2_data/digits/mnistm/train.csv", img_folder="./hw2_data/digits/mnistm/data", image_size=(28, 28))
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

for epoch in range(n_epoch):
    print(f'epoch {epoch}')
    ddpm.train()

    # linear lrate decay
    optim.param_groups[0]['lr'] = lrate*(1-epoch/n_epoch)

    loss_ema = None
    for x, c in tqdm(train_loader):
        optim.zero_grad()
        x = x.to(device)
        c = c.to(device)

        loss = ddpm(x, c)
        loss.backward()

        if loss_ema is None:
            loss_ema = loss.item()
        else:
            loss_ema = 0.95 * loss_ema + 0.05 * loss.item()

        tqdm(train_loader).set_description(f"loss: {loss_ema:.4f}")
        optim.step()
    
    ddpm.eval()
    with torch.no_grad():
        n_sample = 4*n_classes
        x_gen, _ = ddpm.sample(n_sample, (3, 28, 28), device, guide_w=w)

        # append some real images at bottom, order by class also
        x_real = torch.Tensor(x_gen.shape).to(device)
        for k in range(n_classes):
            for j in range(int(n_sample/n_classes)):
                try: 
                    idx = torch.squeeze((c == k).nonzero())[j]
                except:
                    idx = 0
                x_real[k+(j*n_classes)] = x[idx]

        x_all = torch.cat([x_gen, x_real])
        grid = make_grid(x_all*-1 + 1, nrow=10)
        save_image(grid, save_dir + f"image_ep{epoch}_w{w}.png")
        print('saved image at ' + save_dir + f"image_ep{epoch}_w{w}.png")
            
    # optionally save model
    if save_model and epoch == int(n_epoch-1):
        torch.save(ddpm.state_dict(), save_dir + f"model_{epoch}.pth")
        print('saved model at ' + save_dir + f"model_{epoch}.pth")
        