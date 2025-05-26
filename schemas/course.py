from pydantic import BaseModel
from typing import Literal

class SimpleCourseRecommendRequest(BaseModel):
    gender: Literal["male", "female", "unisex"]
    emotion: str
    season: str
    time: str
    latitude: float
    longitude: float

class CourseItem(BaseModel):
    store: str
    address: str
    perfume_name: str
    brand: str
    image_url: str
    distance_km: float
