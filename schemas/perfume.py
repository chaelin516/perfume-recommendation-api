from pydantic import BaseModel
from typing import List

class PerfumeDetail(BaseModel):
    id: int
    name: str
    brand: str
    notes: List[str]
    description: str
    rating: float
