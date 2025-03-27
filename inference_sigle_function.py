import os
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from utils import *
from dataset import *

from model.ediffiqa import EDIFFIQA_CONF


def inference(
    seed=42069,
    device="cuda:0",
    model_name="ediffiqaL",
    dataset_loc="./face/0",
    save_path=".",
    verbose=1,
    params={"batch_size": 128},
):

    # Seed all libraries to ensure consistency between runs.
    seed_all(seed)

    # Load the training FR model and construct the transformation
    conf_loc, weight_loc = EDIFFIQA_CONF[model_name]
    model, trans = construct_full_model(conf_loc)
    model.load_state_dict(torch.load(weight_loc, map_location=torch.device("cpu")))
    model.to(device).eval()

    # Construct the Image Dataloader
    dataset = ImageDataset(dataset_loc, trans)
    dataloader = DataLoader(dataset, **params)

    # Predict quality scores
    quality_scores = {}
    for name_batch, img_batch in tqdm(
        dataloader, desc=" Inference ", disable=not verbose
    ):
        img_batch = img_batch.to(device)
        preds = model(img_batch).detach().squeeze().cpu().numpy()
        quality_scores.update(dict(zip(name_batch, preds)))
    quality_scores = {
        os.path.basename(key): value for key, value in quality_scores.items()
    }  # 改为只保留文件名
    return quality_scores
