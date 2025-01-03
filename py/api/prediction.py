import time
from typing import List
from .client import Client

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

    task_result: Result


class VideoPredictionResponse(BaseModel):
    class Result(BaseModel):
        class VideoDescription(BaseModel):
            id: str = None
            url: str = None
            duration: str = None

        videos: List[VideoDescription] = []

    task_id: str = None

    task_status: str = None

    created_at: str = None

    updated_at: str = None

    task_status_msg: str = None

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

        result = {}
        for key in dir(self):
            value = getattr(self, key)
            if key.startswith('_') or callable(value):
                continue
            if hasattr(value, "to_dict"):
                result[key] = value.to_dict()
            else:
                result[key] = value
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
