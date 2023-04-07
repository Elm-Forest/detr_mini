import requests
import torch
from PIL import Image

from detr_mini import DETR
from predict_utils import transform, rescale_bboxes, plot_results

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
im = Image.open(requests.get(url, stream=True).raw)
torch.set_grad_enabled(False)

detr = DETR(num_classes=91, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6)
state_dict = torch.hub.load_state_dict_from_url(
    url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
    map_location='cpu', check_hash=True)
detr.load_state_dict(state_dict)
detr.eval()
img = transform(im).unsqueeze(0)
assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'
output = detr(img)
logits = output['pred_logits']
bboxes = output['pred_boxes']
probas = logits.softmax(-1)[0, :, :-1]
confidence = 0.75
keep = probas.max(-1).values > confidence
bboxes_scaled = rescale_bboxes(bboxes[0, keep], im.size)
plot_results(im, probas[keep], bboxes_scaled)
