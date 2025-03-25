import time
import os
import ffmpeg
import subprocess
import math

#You can use this node to save full size images through the websocket, the
#images will be sent in exactly the same format as the image previews: as
#binary images on the websocket with a 8 byte header indicating the type
#of binary message (first 4 bytes) and the image format (next 4 bytes).

#Note that no metadata will be put in the images saved with this node.

class SaveCoverAudioVideoKLing:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {
                        "kLing_video_path":("STRING",{"default":""}),
                        "filename_prefix": ("STRING", {"default":"yxkj"}),
                        "cover_frame_num":("INT",{"default":1}),
                        "audio_start_s":("INT",{"default":0}),
                        "audio_path": ("STRING", {"default":""}),
                        "cover_img_path": ("STRING", {"default":""}),
                     },
                }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "save_video"
    OUTPUT_NODE = True
    CATEGORY = "KLingAI"

    def replace_first_frame_with_image(self,input_video, input_image,cover_frame_num=1,filename_prefix="comfyui"):
        try:
            # Generate a unique output path
            dir, file_name = os.path.split(input_video)
            timestamp = time.time()
            filename = f"{filename_prefix}_cover_{timestamp}_{file_name}" 
            output_path = os.path.join(dir,filename)

            # Read video and image
            video = ffmpeg.input(input_video)
            image = ffmpeg.input(input_image)

            # Get video metadata
            probe = ffmpeg.probe(input_video)
            video_info = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            fps = float(video_info['r_frame_rate'].split('/')[0]) / float(video_info['r_frame_rate'].split('/')[1])

            # Set image duration to 1 frame
            cover_frame_num = min(cover_frame_num,fps)
            image_duration = cover_frame_num / fps

            # Scale image to match video resolution
            # Scale image to match video resolution and set SAR
            image = image.filter('scale', video_info['width'], video_info['height']).filter('setsar', '1/1')

            # Trim image to 1 frame and adjust timestamps
            image = image.filter('trim', duration=image_duration).filter('setpts', 'PTS-STARTPTS')

            # Extract audio from the original video (if exists)
            audio = video.audio
            # 检查是否有音频
            streams = probe.get('stream',[])
            has_audio = any(stream.get('codec_type','') == 'audio' for stream in streams)

            # Trim video to remove the first frame and adjust timestamps
            video = video.filter('trim', start=image_duration).filter('setpts', 'PTS-STARTPTS')

            # Concatenate image and video
            combined_video = ffmpeg.concat(image, video, v=1, a=0)

            # Merge video and audio (if audio exists)
            if has_audio:
                output = ffmpeg.output(combined_video, audio, output_path,vsync="2")
            else:
               output = ffmpeg.output(combined_video, output_path,vsync="2")

            # Run ffmpeg command
            output.run(overwrite_output=True)

            return output_path,filename
        except subprocess.CalledProcessError as e:
            print(f"handle_fisson,执行 ffmpeg 命令失败: {e}")
        except FileNotFoundError:
            print("handle_fisson,未找到 ffmpeg,请确保已安装 ffmpeg 并添加到系统环境变量中.")

    def mix_audio_with_video(self,video_path, audio_path, filename_prefix, audio_volume=0.5, audio_start_s=0, original_audio_volume=0.5):
        try:
            timestamp = time.time()
            dir_path, file_name = os.path.split(video_path)
            filename = f"{filename_prefix}_audio_{timestamp}_{file_name}"
            output_path = os.path.join(dir_path, filename)

            # 获取视频时长
            video_info = ffmpeg.probe(video_path)
            video_duration = float(video_info['format']['duration'])

            # 获取外部音频时长和采样率
            audio_info = ffmpeg.probe(audio_path)
            audio_duration = float(audio_info['format']['duration'])

            # 获取音频采样率
            sample_rate = None
            for stream in audio_info['streams']:
                if stream['codec_type'] == 'audio':
                    sample_rate = int(stream['sample_rate'])
                    break

            if not sample_rate:
                print(f"获取外部音频采样率失败,不添加外部音频")
                return video_path, file_name

            # 如果音频起始时间超出音频时长，则直接返回原视频路径
            if audio_start_s >= audio_duration:
                print(f"音频起始时间 {audio_start_s}s 超过音频时长 {audio_duration}s,不添加外部音频")
                return video_path, file_name

            # 加载输入流
            video_input = ffmpeg.input(video_path)
            audio_input = ffmpeg.input(audio_path)

            # 处理外部音频（调整音量并裁剪起始时间）
            adjusted_audio = (
                audio_input.audio
                .filter('atrim', start=audio_start_s)
                .filter('asetpts', 'PTS-STARTPTS')  # 重新对齐时间轴
                .filter('volume', volume=audio_volume)
            )

            # 计算外部音频可用时长
            remaining_audio_duration = max(audio_duration - audio_start_s, 0)

            # 如果外部音频时长短于视频时长，则进行循环
            if remaining_audio_duration < video_duration:
                loops = math.ceil(video_duration / remaining_audio_duration)
                adjusted_audio = adjusted_audio.filter('aloop', loop=loops, size=int(video_duration * sample_rate))

            # 裁剪到视频时长
            adjusted_audio = adjusted_audio.filter('atrim', duration=video_duration)

            # 处理视频原始音频
            video_audio = None
            for stream in video_info['streams']:
                if stream['codec_type'] == 'audio':
                    video_audio = (
                        video_input.audio
                        .filter('volume', original_audio_volume)
                        .filter('atrim', duration=video_duration)
                    )
                    break  

            # 混合音频
            if adjusted_audio and video_audio:
                mixed_audio = ffmpeg.filter([video_audio, adjusted_audio], 'amix', inputs=2)
            else:
                mixed_audio = adjusted_audio if adjusted_audio else video_audio

            # 输出配置
            output = ffmpeg.output(
                video_input.video,
                mixed_audio,
                output_path,
                vcodec='copy',
                acodec='aac',
                strict='experimental'
            )

            output.run(overwrite_output=True)
            return output_path, filename

        except ffmpeg.Error as e:
            print(f"FFmpeg 错误: {e.stderr.decode('utf-8')}")
        except KeyError as e:
            print(f"元数据错误: 缺少必要字段 {e}")
        except Exception as e:
            print(f"未知错误: {str(e)}")

    def save_video(self,kLing_video_path:str,filename_prefix:str,audio_path:str,cover_img_path:str,cover_frame_num:int,audio_start_s:int):
        video_path = kLing_video_path 
        if len(cover_img_path) > 1:
            cover_video,cover_file_name = self.replace_first_frame_with_image(video_path,cover_img_path,cover_frame_num,filename_prefix)
        else:
            cover_video = video_path

        if len(audio_path) > 1:
            cover_audio_video,cover_audio_file_name = self.mix_audio_with_video(cover_video,audio_path,filename_prefix, audio_volume=0.5, audio_start_s=audio_start_s, original_audio_volume=0.5)
        else:
            cover_audio_video = cover_video
            cover_audio_file_name = os.path.split(cover_audio_video)[-1]

        return  {"ui":{"result":(cover_audio_video, cover_audio_file_name)}}
  
