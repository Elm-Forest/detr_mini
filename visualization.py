from collections import namedtuple
from typing import Any

import torch
from tensorboardX import SummaryWriter

from detr import DETR


class ModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_x: torch.Tensor) -> Any:
        data = self.model(input_x)
        if isinstance(data, dict):
            data_named_tuple = namedtuple("ModelEndpoints", sorted(data.keys()))
            data = data_named_tuple(**data)
        elif isinstance(data, list):
            data = tuple(data)
        return data


detr = DETR(num_classes=91, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6)
inputs = torch.randn(1, 3, 800, 1200)
model_wrapper = ModelWrapper(detr)
writer = SummaryWriter('./logs')
# add_graph函数被修改了，原函数不存在strict=False，需要在源码中主动标注
writer.add_graph(detr, inputs, strict=False)
writer.close()
