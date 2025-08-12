from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class Feedback(BaseModel):
    user_id: str
    feedback_type: str
    video_title: str
    content: str

import os
from fastapi import APIRouter
from pydantic import BaseModel
from email.message import EmailMessage
import smtplib
from dotenv import load_dotenv

# .env 파일 로딩
load_dotenv()

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")

router = APIRouter()

class Feedback(BaseModel):
    user_id: str
    feedback_type: str
    video_title: str
    content: str

@router.post("/nova/auth/feedback")
def receive_feedback(feedback: Feedback):
    print(f"Feedback received: {feedback}")

    msg = EmailMessage()
    msg["Subject"] = f"[Nova 피드백] {feedback.video_title}"
    msg["From"] = EMAIL_USER  
    msg["To"] = "merryc1105@gmail.com"  
    msg["Reply-To"] = feedback.user_id  

    msg.set_content(f"""
[피드백 도착]

- 사용자 ID: {feedback.user_id}
- 유형: {feedback.feedback_type}
- 영상 제목: {feedback.video_title}
- 내용:
{feedback.content}
""")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_USER, EMAIL_APP_PASSWORD)
            smtp.send_message(msg)
        return {"message": "피드백이 성공적으로 제출되었습니다."}
    except Exception as e:
        print("메일 전송 실패:", e)
        return {"message": "피드백 제출에 실패했습니다."}
