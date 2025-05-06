from .prediction import Prediction, VideoPredictionResponse
from typing import List

class EffectInput(Prediction):
    model_name: str
    
    mode: str

    image: str

    images: List[str]

    duration: str


class Effects(Prediction):
    effect_scene: str

    input: EffectInput

    def __init__(self):
        super().__init__()
        self._request_method = "POST"
        self._request_path = "/v1/videos/effects"
        self._query_prediction_info_method = "GET"
        self._query_prediction_info_path = "/v1/videos/effects"
        self._response_cls = VideoPredictionResponse
