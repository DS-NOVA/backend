from pydantic import BaseModel

class FeedbackRequest(BaseModel):
    user_id: str
    feedback_type: str
    video_title: str
    content: str  

class FeedbackResponse(BaseModel):
    message: str