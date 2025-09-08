import os
from fastapi import APIRouter
from pydantic import BaseModel, field_validator, EmailStr
from email.message import EmailMessage
import smtplib
from dotenv import load_dotenv
import re
from fastapi.responses import JSONResponse

# .env 파일 로딩
load_dotenv()

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")

router = APIRouter()

class Feedback(BaseModel):
    user_id: str
    user_email: EmailStr | None = None
    feedback_type: str
    video_title: str
    content: str

    @field_validator('user_id', mode='before')
    @classmethod
    def coerce_user_id(cls, v):
        return str(v)
    

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

@router.post("/nova/auth/feedback")
def receive_feedback(feedback: Feedback):
    msg = EmailMessage()
    msg["Subject"] = f"[Nova 피드백] {feedback.video_title}"
    msg["From"] = EMAIL_USER
    msg["To"] = "merryc1105@gmail.com"

    effective_email = None
    if feedback.user_email:
        effective_email = str(feedback.user_email)
    elif EMAIL_RE.match(feedback.user_id):
        effective_email = feedback.user_id

    # Reply-To 설정
    if effective_email:
        msg["Reply-To"] = effective_email

    # ✅ 본문 표기도 동일한 로직으로 표시
    pretty_email = effective_email or "-"

    msg.set_content(
        f"""[피드백 도착]

- 사용자 ID: {feedback.user_id}
- 사용자 이메일: {pretty_email}
- 유형: {feedback.feedback_type}
- 영상 제목: {feedback.video_title}

- 내용:
{feedback.content}
"""
    )
    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(EMAIL_USER, EMAIL_APP_PASSWORD)
            smtp.send_message(msg)
        return {"message": "피드백이 성공적으로 제출되었습니다."}
    except smtplib.SMTPAuthenticationError as e:
        print("메일 전송 실패(인증):", e)
        return JSONResponse(status_code=502, content={"message": "메일 서버 인증 실패(계정/앱 비번 확인)"})
    except Exception as e:
        print("메일 전송 실패:", e)
        return {"message": "피드백 제출에 실패했습니다."}
