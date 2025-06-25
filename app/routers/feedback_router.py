from fastapi import APIRouter, HTTPException, Header
from app.schemas.feedback import FeedbackRequest, FeedbackResponse
import smtplib
# from email.message import EmailMessage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header as EmailHeader
import os
# from app.db.database import get_db

router = APIRouter(
    prefix="/nova/auth",
    tags=["Feedback"]
)

@router.post("/feedback", response_model=FeedbackResponse)
async def send_feedback(
    feedback: FeedbackRequest,
    authorization: str = Header(..., alias="Authorization")
):
    sender_email = os.getenv("EMAIL_USER")
    receiver_email = os.getenv("EMAIL_USER")
    app_password = os.getenv("EMAIL_APP_PASSWORD")

    email_content = f"""[Feedback Notification]
    User ID: {feedback.user_id}
    Video Title: {feedback.video_title}
    Feedback Type: {feedback.feedback_type}
    Content: {feedback.content}
    """

    msg = MIMEMultipart()

    subject = EmailHeader("새로운 피드백이 도착했습니다.", "utf-8")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email
    
    msg.set_charset("utf-8")
    
    # 본문을 UTF-8로 인코딩
    body = MIMEText(email_content, "plain", "utf-8")
    msg.attach(body)


    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender_email, app_password)
            smtp.send_message(msg)
        return FeedbackResponse(message="피드백이 성공적으로 제출되었습니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이메일 전송 실패: {str(e)}")
