import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from app.common.constants import DEVICE, DEFAULT_BATCH_SIZE, DEFAULT_RESIZE_HEIGHT, DEFAULT_RESIZE_WIDTH


@torch.inference_mode()
def get_embeddings(model, data):
    model.to(DEVICE)
    model.eval()

    result = []
    for x, _ in DataLoader(data, batch_size=DEFAULT_BATCH_SIZE):
        x = x.to(DEVICE)
        out = model(x)
        result.extend(out.cpu())

    return torch.stack(result)


def get_transformer(height=DEFAULT_RESIZE_HEIGHT, width=DEFAULT_RESIZE_WIDTH):
    return T.Compose([
        T.Resize((int(height), int(width))),
        T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]),
        T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)
    ])
