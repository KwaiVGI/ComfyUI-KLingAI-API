from .prediction import Prediction, Video2AudioResponse


class Video2Audio(Prediction):
    video_id: str
    video_url: str
    sound_effect_prompt: str
    bgm_prompt: str
    asmr_mode: bool

    def __init__(self):
        super().__init__()
        self._request_method = "POST"
        self._request_path = "/v1/audio/video-to-audio"
        self._query_prediction_info_method = "GET"
        self._query_prediction_info_path = "/v1/audio/video-to-audio"
        self._response_cls = Video2AudioResponse
