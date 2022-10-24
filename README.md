# Self-built DETR object detection network
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Elm-Forest/detr_mini/blob/master/detr_mini.ipynb)\
This repository mainly restores the structure of [Detr Transformer](https://github.com/facebookresearch/detr/blob/main/models/transformer.py)\
The original repository from facebookresearch is available at [detr](https://github.com/facebookresearch/detr/blob/main/models/transformer.py)\
This notebook [detr_mini](https://colab.research.google.com/github/Elm-Forest/detr_mini/blob/master/detr_mini.ipynb) will allow you to more clearly understand how an image with dimension C * H * W works in DETR\
And if you want to see attention visualization, this notebook [enc_atten_vis](https://colab.research.google.com/github/Elm-Forest/detr_mini/blob/master/enc_atten_vis.ipynb) might give you some help

##### Detr Transformer
<div>
	<img src="https://user-images.githubusercontent.com/62285254/194690285-4432101b-fa5e-49e6-93bf-6a1bf160532f.png" width="50%" style="display:inline-block">
</div>

##### Visualization of attention
<figure>
<tr>
	<td><img src="https://user-images.githubusercontent.com/62285254/197464376-2b9f9ca5-283b-4d3d-9307-e04e3030e9b7.png" width="40%"></td>
	<td>
		<img src="https://user-images.githubusercontent.com/62285254/197464504-1353a7a0-f49d-4a52-be99-0198155e2de5.png" width="40%">
		You take any q and you dot it with all the other k's
	</td>
</tr>
</figure>

