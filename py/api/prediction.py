import time
from typing import List
from .client import Client
import torch

try:
    from pydantic import v1 as pydantic
except ImportError:
    import pydantic


class BaseModel(pydantic.BaseModel):
    pass


class PredictionResponse(BaseModel):
    task_id: str = None

    task_status: str = None

    created_at: str = None

    updated_at: str = None


class ImagePredictionResponse(BaseModel):
    class Result(BaseModel):
        class ImageDescription(BaseModel):
            index: str = None
            url: str = None

        images: List[ImageDescription] = []

    task_id: str = None

    task_status: str = None

    task_status_msg: str = None

    created_at: str = None

    updated_at: str = None

    task_result: Result = None


class ImageExpanderPredictionResponse(BaseModel):
    class Result(BaseModel):
        class ImageDescription(BaseModel):
            index: str = None
            url: str = None

        images: List[ImageDescription] = []

    task_id: str = None

    task_status: str = None

    task_status_msg: str = None

    created_at: str = None

    updated_at: str = None

    task_result: Result = None


class VideoPredictionResponse(BaseModel):
    class Result(BaseModel):
        class VideoDescription(BaseModel):
            id: str = None
            url: str = None
            duration: str = None

        videos: List[VideoDescription] = []

    task_id: str = None

    task_status: str = None

    session_id: str = None

    created_at: str = None

    updated_at: str = None

    task_status_msg: str = None

    task_result: Result = None


# TODO Video2Audio / Text2Audio
class AudioResponse(BaseModel):
    class Result(BaseModel):
        class AudiosDescription(BaseModel):
            audio_id: str = None
            url_mp3: str = None
            url_wav: str = None
            duration_mp3: float = None
            duration_wav: float = None

        audios: List[AudiosDescription] = []

    task_id: str = None

    task_status: str = None

    task_status_msg: str = None

    created_at: str = None

    updated_at: str = None

    task_result: Result = None


class Video2AudioResponse(BaseModel):
    class Result(BaseModel):
        class AudiosDescription(BaseModel):
            video_id: str = None
            video_url: str = None
            audio_id: str = None
            url_mp3: str = None
            url_wav: str = None
            duration_mp3: str = None
            duration_wav: str = None

        audios: List[AudiosDescription] = []

    task_id: str = None

    task_status: str = None

    task_status_msg: str = None

    created_at: str = None

    updated_at: str = None

    task_result: Result = None


class MultiImage2VideoResponse(BaseModel):
    class Result(BaseModel):
        class VideosDescription(BaseModel):
            id: str = None
            url: str = None
            duration: str = None

        videos: List[VideosDescription] = []

    task_id: str = None  #

    task_status: str = None

    task_status_msg: str = None

    created_at: str = None

    updated_at: str = None

    task_result: Result = None


class Prediction:

    def __init__(self):
        super().__init__()
        self._request_method = None
        self._request_path = None
        self._query_prediction_info_method = None
        self._query_prediction_info_path = None
        self._response_cls = None

        self._task: PredictionResponse = None
        self._task_info = None

    def to_dict(self):
        def convert_value(value):
            if isinstance(value, torch.Tensor):
                return value.cpu().detach().numpy().tolist()

            if hasattr(value, "to_dict"):
                return value.to_dict()

            if isinstance(value, list):
                return [convert_value(item) for item in value]

            if isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}

            return value

        result = {}
        for key in dir(self):
            value = getattr(self, key)
            if key.startswith('_') or callable(value):
                continue
            result[key] = convert_value(value)
        return result

    def _query_prediction_info(self, client, task_id):
        path = self._query_prediction_info_path + "/" + task_id
        resp = client.request(method=self._query_prediction_info_method, path=path)
        self._task_info = self._response_cls(**resp.get("data"))

    def run(self, client: Client):
        resp = client.request(method=self._request_method, path=self._request_path, json=self.to_dict())
        self._task = PredictionResponse(**resp.get("data"))
        return self.wait(client=client)

    def wait(self, client: Client):
        if self._task is None:
            return None

        while self._task_info is None or self._task_info.task_status in ['submitted', 'processing']:
            time.sleep(client.poll_interval)
            self._query_prediction_info(client, self._task.task_id)

        return self._task_info
