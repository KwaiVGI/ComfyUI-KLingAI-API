from .prediction import Prediction, ImagePredictionResponse


class ImageGenerator(Prediction):
    model_name: str

    prompt: str

    negative_prompt: str

    image: str

    image_reference: str

    image_fidelity: float

    human_fidelity: float

    n: int

    aspect_ratio: str

    def __init__(self):
        super().__init__()
        self._request_method = "POST"
        self._request_path = "/v1/images/generations"
        self._query_prediction_info_method = "GET"
        self._query_prediction_info_path = "/v1/images/generations"
        self._response_cls = ImagePredictionResponse
