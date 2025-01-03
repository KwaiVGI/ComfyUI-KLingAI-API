from .prediction import Prediction, VideoPredictionResponse

class VideoExtend(Prediction):
    video_id: str

    prompt: str

    def __init__(self):
        super().__init__()
        self._request_method = "POST"
        self._request_path = "/v1/videos/video-extend"
        self._query_prediction_info_method = "GET"
        self._query_prediction_info_path = "/v1/videos/video-extend"
        self._response_cls = VideoPredictionResponse

