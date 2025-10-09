import uuid
import json

from py.api import Client, Image2Video, Text2Video, ImageGenerator, CameraControl, CameraControlConfig, \
    KolorsVurtualTryOn,ImageExpander,Video2Audio,Text2Audio,MultiModelVideoEdit
import traceback
import base64
import time


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        base64_encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_encoded


def test_text2video(client):
    text2Video = Text2Video()
    text2Video.model = 'kling-v2-1-master'
    text2Video.prompt = '夕阳下奔跑的骏马'
    text2Video.negative_prompt = '人'
    text2Video.cfg_scale = 0.8
    text2Video.mode = 'std'

    text2Video.camera_control = CameraControl()
    text2Video.camera_control.type = 'left_turn_forward'
    text2Video.aspect_ratio = '16:9'
    text2Video.duration = '5'
    print(text2Video.to_dict())

    ret = text2Video.run(client)
    print(ret)


def test_image2video(client):
    image2Video = Image2Video()
    image2Video.mode = "kling-v1"
    image2Video.image = image_to_base64('')
    image2Video.image_tail = image_to_base64('')
    image2Video.prompt = '夕阳下奔跑的骏马'
    image2Video.negative_prompt = '人'
    image2Video.cfg_scale = 1.0
    image2Video.mode = 'std'
    image2Video.duration = '5'

    ret = image2Video.run(client)
    print(ret)


def test_image_generator(client):
    imgGene = ImageGenerator()
    imgGene.n = 1
    imgGene.image_fidelity = 0.8
    imgGene.model = 'kling-v1'
    imgGene.prompt = '夕阳下奔跑的骏马'
    imgGene.negative_prompt = '人'
    print(imgGene.to_dict())

    ret = imgGene.run(client)
    print(ret)


def test_kolors_vurtual_try_on(client):
    generator = KolorsVurtualTryOn()
    generator.model_name = 'kolors-virtual-try-on-v1'
    generator.human_image = image_to_base64('')
    generator.cloth_image = image_to_base64('')
    print(generator.to_dict())

    ret = generator.run(client)
    print(ret)

def test_image_expander(client):
    generator = ImageExpander()
    generator.image = image_to_base64(r'/Users/gujunchao/comflowy/ComfyUI/custom_nodes/ComfyUI-KLingAI-API/data/test.jpg')

    generator.n=1
    generator.up_expansion_ratio=1
    generator.down_expansion_ratio=0
    generator.right_expansion_ratio=0
    generator.left_expansion_ratio=0

    generator.prompt = "赛博朋克风格"

    # generator.external_task_id = str(uuid.uuid4())

    print(generator.to_dict())

    ret = generator.run(client)
    print(ret)

def test_video2audio(client):
    video2Audio = Video2Audio()

    video2Audio.video_id = "799356726646038574"
    # video2Audio.video_url = "https://v2-kling.kechuangai.com/bs2/upload-ylab-stunt/special-effect/output/KLingMuse_ec914298-7a7c-4b0e-8108-fed97f9f2169/7731253467483968846/output0sfcr.mp4?x-kcdn-pid=112452"
    video2Audio.sound_effect_prompt = "奔放和谐阳光"
    video2Audio.bgm_prompt = "奔放和谐阳光"
    video2Audio.asmr_mode=False

    ret = video2Audio.run(client)
    print(ret)


def test_text2audio(client):
    text2audio = Text2Audio()
    text2audio.prompt = "生成一段高雅的DJ舞曲"
    text2audio.duration = 3.0



    ret = text2audio.run(client)
    print(ret)

def test_multi_model_video_edit(client):
    video_editor = MultiModelVideoEdit()

    video_editor.model_name = "kling-v1-6"
    video_editor.session_id = "112452"
    video_editor.edit_mode = "addition"
    video_editor.prompt = "添加一个小鸟在天空中飞翔"

    ret = video_editor.run(client)
    print(ret)

if __name__ == '__main__':

    try:
        start_time = time.time()

        client = Client(access_key="7a2c8846e7884e06b7d4c27f245ba19f",
                        secret_key="f976de18d9a24d1988202cbee6205040")

        # test_image_generator(client)
        # test_text2video(client)
        # test_image2video(client)
        # test_kolors_vurtual_try_on(client)

        # test_image_expander(client)

        """
        https://v2-kling.kechuangai.com/bs2/upload-ylab-stunt/kling_4d21e145-e4c2-495b-97f1-518a46913268-VTAInferProcessor-1758693702488693744vbo.mp3?x-kcdn-pid=112452',
        url_wav='https://v2-kling.kechuangai.com/bs2/upload-ylab-stunt/kling_fcd19ee3-9058-4ce3-9cd6-5450cbb7a43f-VTAInferProcessor-
        17586937024671786pw9wm.wav?x-kcdn-pid=112452', duration_mp3='5.146', duration_wav='5.08')])
        Elapsed time: 23.31 seconds
        """
        # test_video2audio(client)
        test_text2video(client)
        # test_multi_model_video_edit(client)

        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

    except BaseException as e:

        print(f'exception: {e}')
        print(traceback.format_exc())


