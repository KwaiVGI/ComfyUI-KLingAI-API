from .api import Client, ImageGenerator, Image2Video, Text2Video, CameraControl, CameraControlConfig, KolorsVurtualTryOn, VideoExtend, LipSync, LipSyncInput
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
import time

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
                "model": (["kling-v1","kling-v1-5"],),
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
    RETURN_NAMES = ("images",)

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
        generator.model_name = model
        generator.prompt = prompt
        generator.negative_prompt = negative_prompt
        generator.image = _image_to_base64(image)
        generator.image_fidelity = image_fidelity
        generator.aspect_ratio = aspect_ratio
        generator.n = image_num
        response = generator.run(client)

        images = []
        for image_info in response.task_result.images:
            img = _images2tensor(_decode_image(_fetch_image(image_info.url)))
            print(f'KLing API output: {image_info.url}')
            images.append(img)
        return (torch.cat(images,dim=0),)
    
    @classmethod
    def IS_CHANGED(s,**args):
        return time.time()

class Image2VideoNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("KLING_AI_API_CLIENT",),
                "model": (["kling-v1", "kling-v1-5", "kling-v1-6"],),
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

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("url", "video_id")

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
        generator.model_name = model
        generator.image = _image_to_base64(image)
        generator.image_tail = _image_to_base64(image_tail)
        generator.prompt = prompt
        generator.negative_prompt = negative_prompt
        generator.cfg_scale = cfg_scale
        generator.mode = mode
        generator.duration = duration
        response = generator.run(client)

        for video_info in response.task_result.videos:
            print(f'KLing API output video id: {video_info.id}, url: {video_info.url}')
            return (video_info.url, video_info.id)
        
        return ('', '')
    
    @classmethod
    def IS_CHANGED(s,**args):
        return time.time()


class Text2VideoNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("KLING_AI_API_CLIENT",),
                "model": (["kling-v1", "kling-v1-6"],),
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

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("url", "video_id")

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
        generator.model_name = model
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

        for video_info in response.task_result.videos:
            print(f'KLing API output video id: {video_info.id}, url: {video_info.url}')
            return (video_info.url, video_info.id)
        
        return ('', '')


class KolorsVirtualTryOnNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "client": ("KLING_AI_API_CLIENT",),
                "model_name": (["kolors-virtual-try-on-v1", "kolors-virtual-try-on-v1-5"],),
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

class VideoExtendNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("KLING_AI_API_CLIENT",),
                "video_id": ("STRING", {"multiline": False, "default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "KLingAI"

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("url", "video_id")

    def run(self, client, video_id, prompt):

        generator = VideoExtend()
        generator.video_id = video_id
        generator.prompt = prompt

        response = generator.run(client)

        for video_info in response.task_result.videos:
            print(f'KLing API output video id: {video_info.id}, url: {video_info.url}')
            return (video_info.url, video_info.id)
        
        return ('', '')

class LipSyncTextInputNode:

    audio_types = {
        "阳光少年": "genshin_vindi2",
        "懂事小弟": "zhinen_xuesheng",
        "运动少年": "tiyuxi_xuedi",
        "青春少女": "ai_shatang",
        "温柔小妹": "genshin_klee2",
        "元气少女": "genshin_kirara",
        "阳光男生": "ai_kaiya",
        "幽默小哥": "tiexin_nanyou",
        "文艺小哥": "ai_chenjiahao_712",
        "甜美邻家": "girlfriend_1_speech02",
        "温柔姐姐": "chat1_female_new-3",
        "职场女青": "girlfriend_2_speech02",
        "活泼男童": "cartoon-boy-07",
        "俏皮女童": "cartoon-girl-01",
        "稳重老爸": "ai_huangyaoshi_712",
        "温柔妈妈": "you_pingjing",
        "严肃上司": "ai_laoguowang_712",
        "优雅贵妇": "chengshu_jiejie",
        "慈祥爷爷": "zhuxi_speech02",
        "唠叨爷爷": "uk_oldman3",
        "唠叨奶奶": "laopopo_speech02",
        "和蔼奶奶": "heainainai_speech02",
        "东北老铁": "dongbeilaotie_speech02",
        "重庆小伙": "chongqingxiaohuo_speech02",
        "四川妹子": "chuanmeizi_speech02",
        "潮汕大叔": "chaoshandashu_speech02",
        "台湾男生": "ai_taiwan_man2_speech02",
        "西安掌柜": "xianzhanggui_speech02",
        "天津姐姐": "tianjinjiejie_speech02",
        "Sunny": "genshin_vindi2",
        "Sage": "zhinen_xuesheng",
        "Ace": "AOT",
        "Blossom": "ai_shatang",
        "Peppy": "genshin_klee2",
        "Dove": "genshin_kirara",
        "Shine": "ai_kaiya",
        "Anchor": "oversea_male1",
        "Lyric": "ai_chenjiahao_712",
        "Melody": "girlfriend_4_speech02",
        "Tender": "chat1_female_new-3",
        "Siren": "chat_0407_5-1",
        "Zippy": "cartoon-boy-07",
        "Bud": "uk_boy1",
        "Sprite": "cartoon-girl-01",
        "Candy": "PeppaPig_platform",
        "Beacon": "ai_huangzhong_712",
        "Rock": "ai_huangyaoshi_712",
        "Titan": "ai_laoguowang_712",
        "Grace": "chengshu_jiejie",
        "Helen": "you_pingjing",
        "Lore": "calm_story1",
        "Crag": "uk_man2",
        "Prattle": "laopopo_speech02",
        "Hearth": "heainainai_speech02"
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "voice_id": (list(LipSyncTextInputNode.audio_types.keys()), {"multiline": False, "default": ""}),
                "voice_language": (["zh", "en"], {"multiline": True, "default": "zh"}),
                "voice_speed": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.8,
                    "max": 2.0,
                    "step": 0.1,
                    "round": 0.01,
                    "display": "number",
                    "lazy": True
                })
            }
        }

    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "KLingAI"

    RETURN_TYPES = ("KLING_AI_API_LIPSYNC_INPUT", )
    RETURN_NAMES = ("input", )

    def run(self, text, voice_id, voice_language, voice_speed):
        input = LipSyncInput()
        input.mode = "text2video"
        input.text = text
        if voice_id in LipSyncTextInputNode.audio_types:
            input.voice_id = LipSyncTextInputNode.audio_types[voice_id]
        else:
            input.voice_id = voice_id

        input.voice_language = voice_language
        input.voice_speed = voice_speed
        
        return (input, )
    
class LipSyncAudioInputNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "audio_file": ("STRING", {"multiline": False, "default": ""}),
                "audio_url": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "KLingAI"

    RETURN_TYPES = ("KLING_AI_API_LIPSYNC_INPUT", )
    RETURN_NAMES = ("input", )

    def run(self, audio_file, audio_url):
        input = LipSyncInput()
        input.mode = "audio2video"
        if audio_file is not None and len(audio_file) > 0:
            input.audio_type = "file"
            if os.path.exists(audio_file):
                with open(audio_file, 'rb') as file:
                    file_data = file.read()
                    input.audio_file = base64.b64encode(file_data).decode('utf-8')
            else:
                raise Exception(f"Audio file not found: {audio_file}")

        if audio_url is not None and len(audio_url) > 0:
            input.audio_type = "url"
            input.audio_url = audio_url

        return (input, )

class LipSyncNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("KLING_AI_API_CLIENT", ),
                "video_id": ("STRING", {"multiline": False, "default": ""}),
                "input": ("KLING_AI_API_LIPSYNC_INPUT", )
            },
            "optional": {
                
            }
        }

    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "KLingAI"

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("url", "video_id")

    def run(self, client, video_id, input):

        generator = LipSync()
        input.video_id = video_id
        generator.input = input
        
        response = generator.run(client)

        for video_info in response.task_result.videos:
            print(f'KLing API output video id: {video_info.id}, url: {video_info.url}')
            return (video_info.url, video_info.id)
        
        return ('', '')
