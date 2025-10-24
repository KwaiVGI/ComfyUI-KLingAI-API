from .prediction import Prediction, ImagePredictionResponse


class KolorsVurtualTryOn(Prediction):
    model_name: str

    human_image: str

    cloth_image: str

    def __init__(self):
        super().__init__()
        self._request_method = "POST"
        self._request_path = "/v1/images/kolors-virtual-try-on"
        self._query_prediction_info_method = "GET"
        self._query_prediction_info_path = "/v1/images/kolors-virtual-try-on/"
        self._response_cls = ImagePredictionResponse
