from .prediction import Prediction, VideoPredictionResponse


class Image2Video(Prediction):
    model_name: str

    image: str

    image_tail: str

    prompt: str

    negative_prompt: str

    cfg_scale: float

    mode: str

    duration: str

    def __init__(self):
        super().__init__()
        self._request_method = "POST"
        self._request_path = "/v1/videos/image2video"
        self._query_prediction_info_method = "GET"
        self._query_prediction_info_path = "/v1/videos/image2video"
        self._response_cls = VideoPredictionResponse

