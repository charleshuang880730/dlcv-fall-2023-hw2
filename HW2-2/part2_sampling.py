from PIL import Image
from tqdm.auto import tqdm
import os

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
from UNet import UNet
import argparse

def linear_beta_scheduler(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    return betas

class GaussianDiffusion:
    def __init__(
        self,
        timesteps=1000,
        beta_schedule='linear',
        predefined_noise_folder=None
    ):
        self.timesteps = timesteps
        self.predefined_noise_folder = predefined_noise_folder
        
        if beta_schedule == 'linear':
            betas = linear_beta_scheduler(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas = betas
            
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning
        # of the diffusion chain
        #self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min =1e-20))
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    # get the param of given timestep t
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out
        
    # use ddim to sample
    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        image_size,
        batch_size=8,
        channels=3,
        ddim_timesteps=50,
        ddim_discr_method="uniform",
        ddim_eta=0.0,
        clip_denoised=True,
        use_predefined_noise=False,
        alphas=np.linspace(0, 1, 11)):

        # make ddim timestep sequence
        if ddim_discr_method == 'uniform':
            c = self.timesteps // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c)))
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
        
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
        
        device = next(model.parameters()).device
        # start from pure noise (for each example in the batch)
        
        # For report problem 1 
        if use_predefined_noise and self.predefined_noise_folder:
            sample_img_list = []
            for idx in range(batch_size):
                noise_file = os.path.join(self.predefined_noise_folder, f"{idx:02d}.pt")
                sample_noise = torch.load(noise_file)[:1].to(device)  # 讀取第一個噪聲樣本
                sample_img_list.append(sample_noise)
            sample_img = torch.cat(sample_img_list, dim=0)
        else:
            sample_img = torch.randn((batch_size, channels, image_size, image_size), device=device)
        
        # For report problem 2
        """if use_predefined_noise and self.predefined_noise_folder:
            noise_00 = torch.load(os.path.join(self.predefined_noise_folder, "00.pt"))[:1].to(device)
            noise_01 = torch.load(os.path.join(self.predefined_noise_folder, "01.pt"))[:1].to(device)

            sample_img_list = []
            for alpha in alphas:
                alpha_tensor = torch.tensor(alpha, dtype=torch.float64).to(device)
                theta = torch.acos(torch.sum(noise_00 * noise_01)/(torch.norm(noise_00)*torch.norm(noise_01)))
                linear_noise = (1 - alpha_tensor) * noise_00 + alpha_tensor * noise_01
                # slerp_noise = (torch.sin((1 - alpha_tensor) * theta) / torch.sin(theta)) * noise_00 + (torch.sin(alpha_tensor * theta) / torch.sin(theta)) * noise_01
                sample_img_list.append(linear_noise)
            sample_img = torch.cat(sample_img_list, dim=0)
        else:
            sample_img = torch.randn((batch_size, channels, image_size, image_size), device=device)"""

        for i in tqdm(reversed(range(0, ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)
            
            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, sample_img.shape)
            alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, prev_t, sample_img.shape)
    
            # 2. predict noise using model
            pred_noise = model(sample_img, t)
            
            # 3. get the predicted x_0
            pred_x0 = (sample_img - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)
            
            # 4. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
            
            # 5. compute "direction pointing to x_t" of formula (12)
            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t**2) * pred_noise
            
            # 6. compute x_{t-1} of formula (12)
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(sample_img)

            sample_img = x_prev
            
        return sample_img.cpu().numpy()
    
argparser = argparse.ArgumentParser()
argparser.add_argument("--input", type=str, help="Input file path.")
argparser.add_argument("--output", type=str, default="output.csv", help="Output file path.")
argparser.add_argument("--model", type=str, default="./UNet.pt", help="Model file path.")
args = argparser.parse_args()

gd = GaussianDiffusion(predefined_noise_folder=f"{args.input}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet().to(device)

# 載入訓練好的模型權重
model_path = f"{args.model}"
model.load_state_dict(torch.load(model_path, map_location=device))

# 2. 固定隨機數生成器的種子
torch.manual_seed(4096)

# 3. 建立資料夾
output_folder = f"{args.output}"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

with torch.no_grad():
    ddim_generated_images = gd.ddim_sample(model, 256, batch_size=10, channels=3, ddim_eta=0.0, ddim_timesteps=50, use_predefined_noise=True)

# 保存生成的圖片並按照指定的命名規則命名
    for idx, image in enumerate(ddim_generated_images):
        image_tensor = torch.from_numpy(image).to(device)

        # Apply min-max normalization
        min_val = torch.min(image_tensor)
        max_val = torch.max(image_tensor)
        normalized_tensor = (image_tensor - min_val) / (max_val - min_val)

        filename = f"{idx:02d}.png"
        filepath = os.path.join(output_folder, filename)

        save_image(normalized_tensor, filepath)
