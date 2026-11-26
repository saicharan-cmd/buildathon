from typing import Literal

from pydantic import BaseModel, Field

class DetectionCreate(BaseModel):
    camera_id: str
    plate_text: str
    confidence: float = Field(ge=0, le=1)


class AlertCreate(BaseModel):
    alert_type: str
    message: str
    plate_text: str | None = None
    camera_id: str | None = None
    detection_event_id: str | None = None


class AlertStatusUpdate(BaseModel):
    status: Literal["acknowledged", "resolved"]
