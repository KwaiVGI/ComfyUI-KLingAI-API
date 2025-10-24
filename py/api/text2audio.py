from .prediction import Prediction, AudioResponse


class Text2Audio(Prediction):
    prompt: str

    duration: float

    def __init__(self):
        super().__init__()
        self._request_method = "POST"
        self._request_path = "/v1/audio/text-to-audio"
        self._query_prediction_info_method = "GET"
        self._query_prediction_info_path = "/v1/audio/text-to-audio"
        self._response_cls = AudioResponse
