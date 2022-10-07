import matplotlib.pyplot as plt
import requests
import torch
import torchvision.transforms as T
from PIL import Image

from detr import DETR

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
url = 'https://img2.baidu.com/it/u=2806373923,1277370447&fm=253&fmt=auto&app=138&f=JPEG?w=800&h=500'
url = 'https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fimg.aiimg.com%2Fuploads%2Fallimg%2F191103%2F263915-191103095J1.jpg&refer=http%3A%2F%2Fimg.aiimg.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1667668266&t=d99f0afd798f60dbe90631bdce470fec'
url = 'https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fx0.ifengimg.com%2Fres%2F2020%2FFBA492FF7CF714F2D6FC05D84567824350E61AB4_size153_w1024_h1007.jpeg&refer=http%3A%2F%2Fx0.ifengimg.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1667668266&t=9849ba0a334a1621e99787466c77902f'
url = 'https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fpic.ruiwen.com%2Fallimg%2F1709%2F59c44a5815a6963922.jpg&refer=http%3A%2F%2Fpic.ruiwen.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1667668266&t=16e553a6abab4ed643b66dc6e0a42013'
url = 'https://img1.baidu.com/it/u=17966390,1110814961&fm=253&fmt=auto&app=120&f=JPEG?w=640&h=428'
im = Image.open(requests.get(url, stream=True).raw)
torch.set_grad_enabled(False)
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
img = transform(im).unsqueeze(0)
assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()


detr = DETR(num_classes=91, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6)
state_dict = torch.hub.load_state_dict_from_url(
    url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
    map_location='cpu', check_hash=True)
detr.load_state_dict(state_dict)
detr.eval()
output = detr(img)
logits = output['pred_logits']
bboxes = output['pred_boxes']
probas = logits.softmax(-1)[0, :, :-1]
confidence = 0.8
keep = probas.max(-1).values > confidence
bboxes_scaled = rescale_bboxes(bboxes[0, keep], im.size)
plot_results(im, probas[keep], bboxes_scaled)
