from pydantic import BaseModel, Field
from typing import Optional, Dict


class Prediction(BaseModel):
    asset_id: str
    prediction_price: Optional[float] = None
