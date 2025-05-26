from pydantic import BaseModel
from typing import Any

class BaseResponse(BaseModel):
    message: str
    result: Any = None
