import os
import torch
from torchvision.utils import save_image, make_grid
from Conditional_UNet import ContextUnet
from ddpm import DDPM
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--output", type=str, default="output.csv", help="Output file path.")
args = argparser.parse_args()

# 1. 建立DDPM模型實例
n_T = 400
n_classes = 10
n_feat = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
ddpm.to(device)

# 載入訓練好的模型權重
model_path = './Part1_diffusion_best.ckpt'
ddpm.load_state_dict(torch.load(model_path, map_location=device))
ddpm.eval()

# 2. 固定隨機數生成器的種子
torch.manual_seed(4096)

# 3. 建立資料夾
output_folder = f"{args.output}"
# 4. 生成圖片
image_size = (3, 28, 28)
total_images_per_digit = 100
n_sample = 10 * total_images_per_digit
ws_test = [2.0]
batch_size = 100  

# 使用模型生成條件性的數字圖片
with torch.no_grad():
    generated_images, _ = ddpm.sample(n_sample, image_size, device, guide_w=ws_test[0]) # assuming ws_test is correctly defined

# 5. 儲存圖片
# 儲存圖片
all_images = []
for i in range(total_images_per_digit):
    for digit in range(10):
        image = generated_images[i * 10 + digit]
        all_images.append(image)
        filename = os.path.join(output_folder, f"{digit}_{i+1:03}.png")
        save_image(image, filename)

# 轉換成網格圖片
"""grid_image = make_grid(all_images, nrow=10) # 10 images per row
filename = os.path.join(output_folder, "grid_image.png")
save_image(grid_image, filename)"""

"""with torch.no_grad():
    _, x_i_store = ddpm.sample(10, image_size, device, guide_w=ws_test[0])

# 從 x_i_store 中選擇第一個 "0" 的圖片
selected_images = [x_i_store[i][0] for i in range(len(x_i_store)) if i % 5 == 0]

# 保存六張圖片
for idx, image in enumerate(selected_images):
    image_tensor = torch.from_numpy(image).to(device)
    filename = os.path.join("./HW2-1/timestep_output", f"time_step_{idx}.png")
    save_image(image_tensor, filename)"""