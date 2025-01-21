from .prediction import Prediction, VideoPredictionResponse


class CameraControlConfig(Prediction):
    horizontal: float
    vertical: float
    pan: float
    tilt: float
    roll: float
    zoom: float


class CameraControl(Prediction):
    type: float
    config: CameraControlConfig


class Text2Video(Prediction):
    model_name: str

    prompt: str

    negative_prompt: str

    cfg_scale: float

    mode: str

    camera_control: CameraControl

    aspect_ratio: str

    duration: str

    def __init__(self):
        super().__init__()
        self._request_method = "POST"
        self._request_path = "/v1/videos/text2video"
        self._query_prediction_info_method = "GET"
        self._query_prediction_info_path = "/v1/videos/text2video"
        self._response_cls = VideoPredictionResponse

