from .prediction import Prediction, VideoPredictionResponse


class LipSyncInput(Prediction):
    video_id: str
    video_url: str
    mode: str
    text: str
    voice_id: str
    voice_language: str
    voice_speed: float
    audio_type: str
    audio_file: str
    audio_url: str


class LipSync(Prediction):
    input: LipSyncInput

    def __init__(self):
        super().__init__()
        self._request_method = "POST"
        self._request_path = "/v1/videos/lip-sync"
        self._query_prediction_info_method = "GET"
        self._query_prediction_info_path = "/v1/videos/lip-sync"
        self._response_cls = VideoPredictionResponse
