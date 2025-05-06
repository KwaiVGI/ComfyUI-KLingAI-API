# ComfyUI-KLingAI-API

This is a custom node for ComfyUI that allows you to use the KLing AI API directly in ComfyUI. KLing AI API is based on top of [KLing AI](https://app.klingai.com/global/dev/document-api/quickStart/productIntroduction/overview). For more information, see [KLing AI API Documentation](https://docs.qingque.cn/d/home/eZQArO-0RpjbQMpf5DPa-w8Rp?identityId=1oEER8VjdS8#section=h.wxrj3t7cbpg3).


## Requirements
Before using this node, you need to have [a KLing AI API key](https://docs.qingque.cn/d/home/eZQA6m4cRjTB1BBiE5eJ4lyvL?identityId=1oEER8VjdS8). 

## Installation

### Installing manually

1. Navigate to the `ComfyUI/custom_nodes` directory.

2. Clone this repository: `git clone https://github.com/KwaiVGI/ComfyUI-KLingAI-API`
  
3. Install the dependencies:
  - Windows (ComfyUI portable): `python -m pip install -r ComfyUI\custom_nodes\ComfyUI-KLingAI-API\requirements.txt`
  - Linux or MacOS: `cd ComfyUI-KLingAI-API && pip install -r requirements.txt`

4. If you don't want to expose your key, you can add it into the `config.ini` file and keep it empty in the node.

5. Start ComfyUI and enjoy using the KLing AI API node!

## Nodes

### Client

This node is used to create a KLing AI client.

### Image Generator

This node is used to generate an image given a text prompt.
<p align="center">
  <img src="./examples/image_generation.png" alt="image_generation">
</p>


### Text2Video

This node is used to generate a video given a text prompt.
<p align="center">
  <img src="./examples/text2video.png" alt="text2video">
</p>

### Image2Video

This node is used to generate a video given an image.
<p align="center">
  <img src="./examples/image2video.png" alt="image2video">
</p>

### Kolors Virtual Try-On

This node is used to display the try-on effect.
<p align="center">
  <img src="./examples/kolors_virtual_try_on.png" alt="text2video">
</p>

### Video Extend
This node is used to extend a video.
<p align="center">
  <img src="./examples/video_extend.png" alt="video_extend">
</p>

### Lip Sync
This node is used to generate a lip sync video.
<p align="center">
  <img src="./examples/lip_sync.png" alt="lip_sync">
</p>

### Effects
You can achieve different special effects based on the effect_scene.
<p align="center">
  <img src="./examples/video_effects.png" alt="lip_sync">
</p>

## Pricing

For pricing, follow [KLing AI Pricing](https://klingai.com/dev-center).