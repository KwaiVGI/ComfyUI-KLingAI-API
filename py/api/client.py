import time
import jwt
import httpx
from .exceptions import KLingAPIError
from enum import Enum

def _raise_for_status(resp: httpx.Response) -> None:
    if resp.status_code != 200 or 'data' not in resp.json():
        raise KLingAPIError.from_response(resp)


class ApiLocation(Enum):
    CHINA = "https://api-beijing.klingai.com"
    GLOBAL = "https://api.klingai.com"


class Client:
    __token = None
    __client = None

    def __init__(self, access_key, secret_key, in_china=True, timeout=5, poll_interval=1.0, ttl=1800):
        super().__init__()
        self._access_key = access_key
        self._secret_key = secret_key
        self._timeout = timeout
        self._create_time = None
        self._ttl = ttl
        self._area = ApiLocation.CHINA if in_china else ApiLocation.GLOBAL

        self.poll_interval = poll_interval

    def request(self, method: str, path: str, **kwargs) -> httpx.Response:
        resp = self._client.request(method, path, **kwargs)
        _raise_for_status(resp)
        return resp.json()

    @property
    def _is_expired(self):
        if self._create_time is None:
            return True
        return time.time() - self._create_time > self._ttl

    @property
    def _token(self):
        self._create_time = time.time()
        headers = {
            "alg": "HS256",
            "typ": "JWT"
        }
        payload = {
            "iss": self._access_key,
            "exp": int(self._create_time) + self._ttl,
            "nbf": int(self._create_time) - 5
        }
        self.__token = jwt.encode(payload, self._secret_key, headers=headers)
        print(f'create token: {self._create_time}')
        return self.__token

    @property
    def _client(self) -> httpx.Client:
        if self._is_expired:
            base_url = str(self._area.value)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._token}"
            }
            timeout = httpx.Timeout(self._timeout)
            self.__client = httpx.Client(base_url=base_url, headers=headers, timeout=timeout)
        return self.__client
