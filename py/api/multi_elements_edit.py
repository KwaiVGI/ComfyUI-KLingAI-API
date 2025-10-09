"""
    多模态视频编辑
"""
from array import array
from .prediction import Prediction, VideoPredictionResponse

class MultiModelVideoEdit(Prediction):
    model_name: str

    session_id: str

    edit_mode: str

    image_list: array

    prompt: str

    negative_prompt: str

    mode = str

    duration:str


    def __init__(self):
        super().__init__()
        self._request_method = "POST"
        self._request_path = "/v1/videos/multi-elements"
        self._query_prediction_info_method = "GET"
        self._query_prediction_info_path = "/v1/videos/multi-elements"
        self._response_cls = VideoPredictionResponse

