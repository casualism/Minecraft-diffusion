from ViT import ViT
from torch.hub import load_state_dict_from_url
dependencies = ['torch', 'math']


def general_model():
    checkpoint = 'https://github.com/casualism/minecraft-diffusion/releases/download/v1/better_model.pt'
    model = ViT(1024, 1000)
    return model.load_state_dict(load_state_dict_from_url(checkpoint, progress=True))


def better_model():
    checkpoint = 'https://github.com/casualism/minecraft-diffusion/releases/download/v1/better_model.pt'
    model = ViT(1024, 1000)
    return model.load_state_dict(load_state_dict_from_url(checkpoint, progress=True))