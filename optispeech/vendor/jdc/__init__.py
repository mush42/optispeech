import os
from pathlib import Path

import torch

from .model import JDCNet


DEFAULT_MODEL_PATH = os.fspath(Path(__file__).parent.joinpath("bst.t7"))


def load_F0_model(model_path=DEFAULT_MODEL_PATH):
    # load F0 model
    F0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load(model_path, map_location='cpu')['net']
    F0_model.load_state_dict(params)
    _ = F0_model.train()
    
    return F0_model
