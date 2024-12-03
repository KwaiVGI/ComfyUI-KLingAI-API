from .api import Client, ImageGenerator, Image2Video, Text2Video, CameraControl, CameraControlConfig, KolorsVurtualTryOn
import base64
import io
import os
import re
import numpy
import PIL
import requests
import torch
from collections.abc import Iterable
import configparser
import folder_paths

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
config_path = os.path.join(parent_dir, 'config.ini')
config = configparser.ConfigParser()
config.read(config_path)


def _fetch_image(url, stream=True):
    return requests.get(url, stream=stream).content


def _tensor2images(tensor):
    np_imgs = numpy.clip(tensor.cpu().numpy() * 255.0, 0.0, 255.0).astype(numpy.uint8)
    return [PIL.Image.fromarray(np_img) for np_img in np_imgs]


def _images2tensor(images):
    if isinstance(images, Iterable):
        return torch.stack([torch.from_numpy(numpy.array(image)).float() / 255.0 for image in images])
    return torch.from_numpy(numpy.array(images)).unsqueeze(0).float() / 255.0


def _decode_image(data_bytes, rtn_mask=False):
    with io.BytesIO(data_bytes) as bytes_io:
        img = PIL.Image.open(bytes_io)
        if not rtn_mask:
            img = img.convert('RGB')
        elif 'A' in img.getbands():
            img = img.getchannel('A')
        else:
            img = None
    return img


def _encode_image(img, mask=None):
    if mask is not None:
        img = img.copy()
        img.putalpha(mask)
    with io.BytesIO() as bytes_io:
        if mask is not None:
            img.save(bytes_io, format='PNG')
        else:
            img.save(bytes_io, format='JPEG')
        data_bytes = bytes_io.getvalue()
    return data_bytes


def _image_to_base64(image):
    if image is None:
        return None
    return base64.b64encode(_encode_image(_tensor2images(image)[0])).decode("utf-8")


class KLingAIAPIClient:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "access_key": ("STRING", {"multiline": False, "default": ""}),
                "secret_key": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("KLING_AI_API_CLIENT",)
    RETURN_NAMES = ("client",)

    FUNCTION = "create_client"

    OUTPUT_NODE = True

    CATEGORY = "KLingAI"

    def create_client(self, access_key, secret_key):

        if access_key == "" or secret_key == "":
            try:
                klingai_api_access_key = config['API']['KLINGAI_API_ACCESS_KEY']
                klingai_api_scerct_key = config['API']['KLINGAI_API_SECRET_KEY']
                if klingai_api_access_key == '':
                    raise ValueError('ACCESS_KEY is empty')
                if klingai_api_scerct_key == '':
                    raise ValueError('SECRET_KEY is empty')

            except KeyError:
                raise ValueError('unable to find ACCESS_KEY or SECRET_KEY in config.ini')

            client = Client(klingai_api_access_key, klingai_api_scerct_key)
        else:
            client = Client(access_key, secret_key)

        return (client,)


class ImageGeneratorNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("KLING_AI_API_CLIENT",),
                "model": (["kling-v1"],),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "image": ("IMAGE",),
                "image_fidelity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "round": 0.01,
                    "display": "number",
                    "lazy": True
                }),
                "image_num": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 9,
                    "step": 1,
                    "display": "number",
                    "lazy": True
                }),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4", "3:2", "2:3"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "generate"

    OUTPUT_NODE = False

    CATEGORY = "KLingAI"

    def generate(self,
                 client,
                 model,
                 prompt,
                 negative_prompt=None,
                 image=None,
                 image_fidelity=None,
                 image_num=None,
                 aspect_ratio=None):
        generator = ImageGenerator()
        generator.model = model
        generator.prompt = prompt
        generator.negative_prompt = negative_prompt
        generator.image = _image_to_base64(image)
        generator.image_fidelity = image_fidelity
        generator.aspect_ratio = aspect_ratio
        generator.n = image_num
        response = generator.run(client)

        for image_info in response.task_result.images:
            img = _images2tensor(_decode_image(_fetch_image(image_info.url)))
            print(f'KLing API output: {image_info.url}')
            return (img,)


class Image2VideoNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("KLING_AI_API_CLIENT",),
                "model": (["kling-v1", "kling-v1-5"],),
                "image": ("IMAGE",),

            },
            "optional": {
                "image_tail": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "cfg_scale": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "round": 0.01,
                    "display": "number",
                    "lazy": True
                }),
                "mode": (["std", "pro"],),
                "duration": (["5", "10"],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("url",)

    FUNCTION = "generate"

    OUTPUT_NODE = False

    CATEGORY = "KLingAI"

    def generate(self,
                 client,
                 model,
                 image,
                 image_tail=None,
                 prompt=None,
                 negative_prompt=None,
                 cfg_scale=None,
                 mode=None,
                 duration=None):
        generator = Image2Video()
        generator.model = model
        generator.image = _image_to_base64(image)
        generator.image_tail = _image_to_base64(image_tail)
        generator.prompt = prompt
        generator.negative_prompt = negative_prompt
        generator.cfg_scale = cfg_scale
        generator.mode = mode
        generator.duration = duration
        response = generator.run(client)

        result = []
        for video_info in response.task_result.videos:
            result.append(video_info.url)
        print(f'KLing API output: {result}')
        return (result,)


class Text2VideoNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("KLING_AI_API_CLIENT",),
                "model": (["kling-v1"],),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "cfg_scale": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "round": 0.01,
                    "display": "number",
                    "lazy": True
                }),
                "mode": (["std", "pro"],),
                "aspect_ratio": (["16:9", "9:16", "1:1"],),
                "duration": (["5", "10"],),
                "camera_control_type": (
                    ["simple", "down_back", "forward_up", "right_turn_forward", "left_turn_forward"],),
                "camera_control_config": (["horizontal", "vertical", "pan", "tilt", "roll", "zoom"],),
                "camera_control_value": ("FLOAT", {
                    "default": 0.5,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 1.0,
                    "round": 1.0,
                    "display": "number",
                    "lazy": True
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("url",)

    FUNCTION = "generate"

    OUTPUT_NODE = False

    CATEGORY = "KLingAI"

    def generate(self,
                 client,
                 model,
                 prompt,
                 negative_prompt=None,
                 cfg_scale=None,
                 mode=None,
                 aspect_ratio=None,
                 duration=None,
                 camera_control_type=None,
                 camera_control_config=None,
                 camera_control_value=None):

        generator = Text2Video()
        generator.model = model
        generator.prompt = prompt
        generator.prompt = prompt
        generator.negative_prompt = negative_prompt
        generator.cfg_scale = cfg_scale
        generator.mode = mode
        generator.aspect_ratio = aspect_ratio
        generator.duration = duration

        generator.camera_control = CameraControl()
        generator.camera_control.type = camera_control_type

        if generator.camera_control.type == "simple":
            generator.camera_control.config = CameraControlConfig()
            if camera_control_config == "horizontal":
                generator.camera_control.config.horizontal = camera_control_value
            if camera_control_config == "vertical":
                generator.camera_control.config.vertical = camera_control_value
            if camera_control_config == "pan":
                generator.camera_control.config.pan = camera_control_value
            if camera_control_config == "tilt":
                generator.camera_control.config.tilt = camera_control_value
            if camera_control_config == "roll":
                generator.camera_control.config.roll = camera_control_value
            if camera_control_config == "zoom":
                generator.camera_control.config.zoom = camera_control_value

        response = generator.run(client)

        result = []
        for video_info in response.task_result.videos:
            result.append(video_info.url)
        print(f'KLing API output: {result}')
        return (result,)


class KolorsVirtualTryOnNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("KLING_AI_API_CLIENT",),
                "model_name": (["kolors-virtual-try-on-v1"],),
                "human_image": ("IMAGE",),
                "cloth_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "generate"

    OUTPUT_NODE = False

    CATEGORY = "KLingAI"

    def generate(self,
                 client,
                 model_name,
                 human_image,
                 cloth_image=None):
        generator = KolorsVurtualTryOn()
        generator.model_name = model_name
        generator.human_image = _image_to_base64(human_image)
        generator.cloth_image = _image_to_base64(cloth_image)

        response = generator.run(client)

        for image_info in response.task_result.images:
            img = _images2tensor(_decode_image(_fetch_image(image_info.url)))
            print(f'KLing API output: {image_info.url}')
            return (img,)


class PreviewVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_url": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "KLingAI"}),
                "save_output": ("BOOLEAN", {"default": True}),
            }
        }

    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "KLingAI"

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)

    def run(self, video_url, filename_prefix, save_output):
        if not save_output:
            return {"ui": {"video_url": [video_url]}, "result": ('', )}
        
        output_dir = folder_paths.get_output_directory()
        (
            full_output_folder,
            filename,
            _,
            _,
            _,
        ) = folder_paths.get_save_image_path(filename_prefix, output_dir)

        max_counter = 0

        matcher = re.compile(f"{re.escape(filename)}_(\\d+)\\D*\\..+", re.IGNORECASE)
        for existing_file in os.listdir(full_output_folder):
            match = matcher.fullmatch(existing_file)
            if match:
                file_counter = int(match.group(1))
                if file_counter > max_counter:
                    max_counter = file_counter

        counter = max_counter + 1
        file = f"{filename}_{counter:05}.mp4"
        file_path = os.path.join(full_output_folder, file)

        if type(video_url) == list:
            video_url = video_url[0]
        open(file_path, "wb").write(_fetch_image(video_url))            

        return {"ui": {"video_url": [video_url]}, "result": (file_path, )}
