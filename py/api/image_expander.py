from .prediction import Prediction, ImageExpanderPredictionResponse


class ImageExpander(Prediction):

    image: str

    up_expansion_ratio: float

    down_expansion_ratio:float

    left_expansion_ratio:float

    right_expansion_ratio:float

    prompt:str

    n:int

    # external_task_id:str

    def __init__(self):
        super().__init__()
        self._request_method = "POST"
        self._request_path = "/v1/images/editing/expand"
        self._query_prediction_info_method = "GET"
        self._query_prediction_info_path = "/v1/images/editing/expand"
        self._response_cls = ImageExpanderPredictionResponse
