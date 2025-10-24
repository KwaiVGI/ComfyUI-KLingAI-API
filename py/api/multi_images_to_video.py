from array import array
from .prediction import Prediction, MultiImage2VideoResponse


class MultiImages2Video(Prediction):
    model_name: str

    image_list: array

    image_tail: str

    prompt: str

    negative_prompt: str

    mode: str

    duration: str

    aspect_ratio: str

    def __init__(self):
        super().__init__()
        self._request_method = "POST"
        self._request_path = "/v1/videos/multi-image2video"
        self._query_prediction_info_method = "GET"
        self._query_prediction_info_path = "/v1/videos/multi-image2video"
        self._response_cls = MultiImage2VideoResponse
