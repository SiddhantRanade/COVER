import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import argparse
import os
import pickle as pkl

import decord
import numpy as np
import yaml
from tqdm import tqdm
import json

from torch.nn.parallel import DistributedDataParallel as DDP

from cover.datasets import (
    UnifiedFrameSampler,
    ViewDecompositionDataset,
    spatial_temporal_view_decomposition,
)
from cover.models import COVER

mean, std = (
    torch.FloatTensor([123.675, 116.28, 103.53]),
    torch.FloatTensor([58.395, 57.12, 57.375]),
)

mean_clip, std_clip = (
    torch.FloatTensor([122.77, 116.75, 104.09]),
    torch.FloatTensor([68.50, 66.63, 70.32])
)


def fuse_results(results: list):
    x = (results[0] + results[1] + results[2])
    return {
        "semantic" : float(results[0]),
        "technical": float(results[1]),
        "aesthetic": float(results[2]),
        "overall"  : float(x),
    }

def setup_distributed():
    # Initialize the process group
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_distributed():
    dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--opt", type=str, default="./cover.yml", help="the option file")
    parser.add_argument("-i", "--input_video_dir", type=str, default="./demo", help="the input video dir")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    out_file = os.path.join(args.input_video_dir, 'quality.json')
    # Set up distributed training
    setup_distributed()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)

    ### Load COVER
    evaluator = COVER(**opt["model"]["args"]).to(device)
    state_dict = torch.load(opt["test_load_path"], map_location=device)
    
    # set strict=False here to avoid error of missing
    # weight of prompt_learner in clip-iqa+, cross-gate
    evaluator.load_state_dict(state_dict['state_dict'], strict=False)

    evaluator = DDP(evaluator, device_ids=[int(os.environ["LOCAL_RANK"])])

    dopt = opt["data"]["val-l1080p"]["args"]

    dopt["anno_file"] = None
    dopt["data_prefix"] = args.input_video_dir

    # Load existing results if any

    dataset = ViewDecompositionDataset(dopt)

    # Add distributed sampler
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=opt["num_workers"],
        pin_memory=True,
        sampler=sampler
    )

    sample_types = ["semantic", "technical", "aesthetic"]

    for i, data in enumerate(pbar := tqdm(dataloader, desc="Testing", disable=dist.get_rank() != 0)):
        # video_name = data["name"][0].split("/")[-1]
        video_name = data["name"][0].replace(args.input_video_dir, "")
        if len(data.keys()) == 1:
            continue

        pbar.set_description(f"Processing {video_name}")

        video = {}
        try:
            for key in sample_types:
                if key in data:
                    video[key] = data[key].to(device)
                    b, c, t, h, w = video[key].shape
                    video[key] = (
                        video[key]
                        .reshape(
                            b, c, data["num_clips"][key], t // data["num_clips"][key], h, w
                        )
                        .permute(0, 2, 1, 3, 4, 5)
                        .reshape(
                            b * data["num_clips"][key], c, t // data["num_clips"][key], h, w
                        )
                    )

            with torch.no_grad():
                results = evaluator(video, reduce_scores=False)
                results = [np.mean(l.cpu().numpy()) for l in results]
            rescaled_results = fuse_results(results)
            json_name = data["json_name"][0]
            with open(json_name, 'w') as f:
                json.dump(rescaled_results, f, indent=2)
        except Exception as e:
            with open(os.path.join(args.input_video_dir, 'error.log'), 'a') as f:
                f.write(f"Error in {video_name}: {e}\n")
                f.write(f"Results: {results}\n")

    cleanup_distributed()
