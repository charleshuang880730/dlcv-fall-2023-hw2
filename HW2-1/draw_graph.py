from torchview import draw_graph
from Conditional_UNet import ContextUnet
from ddpm import DDPM
import torch

n_epoch = 20
batch_size = 256
n_T = 400 # 500
device = "cuda:0"
n_classes = 10
n_feat = 128 # 128 ok, 256 better (but slower)
ws_test = [0.0, 0.5, 2.0] # strength of generative guidance
batch_size = 2

if __name__ == '__main__':
    model = model = ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes)
    input = torch.randn(batch_size, 3, 28, 28)
    label_mask = torch.bernoulli(torch.zeros((batch_size, )) + 0.1)
    n_step = torch.randn(batch_size)
    label = torch.randint(low=0, high=10, size=(batch_size, ))
    model_graph = draw_graph(
        model,
        input_data = [input, label, n_step, label_mask],
        graph_name = 'try',
        save_graph = True,
        directory = '.',
        filename = 'arch',
        roll = True
    )
    # model_graph.visual_graph