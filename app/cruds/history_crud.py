from sqlalchemy.orm import Session
from sqlalchemy.orm import joinedload
from app.models import History, Video, VideoDetails

# 히스토리 조회
def get_history_by_video(db: Session, video_id: int):
    results = (
        db.query(
            History,
            Video.video_name,
            Video.upload_date,
        )
        .join(Video, History.video_id == Video.id)
        .outerjoin(VideoDetails, Video.id == VideoDetails.video_id)
        .filter(History.video_id == video_id)
        .all()
    )
    history_list = []
    for history, video_name, upload_date in results:
        history_list.append({
            "video_id":history.video_id,
            "video_title": video_name,
            "upload_date": upload_date,
            "timelines": None,  # 나중에 필요하면 추가
        })
    return history_list

# 히스토리 삭제
def delete_history_by_video(db: Session, video_id: int):
    deleted = db.query(History).filter(History.video_id == video_id).delete()
    db.commit()
    return deleted