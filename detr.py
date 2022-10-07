import copy
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50

warnings.filterwarnings('ignore')


class DETR(nn.Module):

    def __init__(self, num_classes, hidden_dim, nheads, num_encoder_layers, num_decoder_layers) -> None:
        super().__init__()
        self.backbone = resnet50(pretrained=True)
        del self.backbone.fc
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        h = self.conv(x)
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        h = self.transformer(pos, 0.1 * h.flatten(2).permute(2, 0, 1), self.query_pos).transpose(0, 1)
        return {'pred_logits': self.linear_class(h),
                'pred_boxes': self.linear_bbox(h).sigmoid()}


class Transformer(nn.Module):
    def __init__(self, d_model=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6) -> None:
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nheads, dim_feedforward=2048, drop_out=0.1)
        decoder_layer = TransformerDecoderLayer(d_model, nheads, dim_feedforward=2048, drop_out=0.1)
        self.encoder = TransformerEncoder(d_model, encoder_layer, num_encoder_layers)
        self.decoder = TransformerDecoder(d_model, decoder_layer, num_decoder_layers)

    def forward(self, pos, h, query_pos):
        query_embed = query_pos.unsqueeze(1).repeat(1, 1, 1)
        tgt = torch.zeros_like(query_embed)
        mem = self.encoder(pos, h)
        tgt = self.decoder(tgt, query_embed, mem, pos)
        return tgt


class TransformerEncoder(nn.Module):

    def __init__(self, d_model, encoder_layer, num_encoder_layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_encoder_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, pos, h):
        for layer in self.layers:
            h = layer(pos, h)
        output = self.norm(h)
        return output


class TransformerDecoder(nn.Module):

    def __init__(self, d_model, decoder_layer, num_decoder_layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_decoder_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, query_embed, mem, pos):
        for layer in self.layers:
            tgt = layer(tgt, query_embed, mem, pos)
        output = self.norm(tgt)
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=2048, drop_out=0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=drop_out)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(drop_out)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_out)
        self.dropout2 = nn.Dropout(drop_out)
        self.activation = F.relu

    def forward(self, pos, h):
        q = k = pos + h
        src2 = self.self_attn(q, k, value=h, attn_mask=None, key_padding_mask=None)[0]
        # src2 = self.self_attn(q, k, value=q, attn_mask=None, key_padding_mask=None)[0]
        src = h + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_feedforward=2048, drop_out=0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=drop_out)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=drop_out)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(drop_out)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_out)
        self.dropout2 = nn.Dropout(drop_out)
        self.dropout3 = nn.Dropout(drop_out)
        self.activation = F.relu

    def forward(self, tgt, query_embed, mem, pos):
        q = k = tgt + query_embed
        # q = k = query_embed
        # tgt2 = self.self_attn(q, k, value=query_embed, attn_mask=None, key_padding_mask=None)[0]
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=None, key_padding_mask=None)[0]
        tgt = tgt + self.dropout1(tgt2)
        # tgt = query_embed + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=tgt + query_embed,
                                   key=mem + pos,
                                   value=mem, attn_mask=None, key_padding_mask=None)[0]
        # tgt2 = self.multihead_attn(query=query_embed,
        #                            key=src,
        #                            value=src, attn_mask=None, key_padding_mask=None)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
