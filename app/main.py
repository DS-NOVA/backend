from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.database import SessionLocal
from sqlalchemy import text
from app.db.database import SessionLocal, Base, engine, init_models 
from app.routers import user_router
from app.routers import upload_router
from fastapi.staticfiles import StaticFiles
from app.routers import history_router 
from app.routers import feedback_router

from app.routers import pipeline_predict
from app.db.mongo import init_mongo
from app.cruds.interp_client import InterpClient
from dotenv import load_dotenv
import os


app = FastAPI()

load_dotenv()
model_server_url = os.getenv("INTERP_MODEL_SERVER_URL")
interp_client = InterpClient(model_server_url)

origins = [
    "http://127.0.0.1:5501",
    "http://localhost:5501",
    "http://127.0.0.1:8000", 
    "http://localhost:8000",
    "http://localhost:5502",
    "http://127.0.0.1:5502"
]

app.include_router(user_router.router)
app.include_router(upload_router.router)
app.include_router(history_router.router)
app.include_router(feedback_router.router)
app.mount("/static", StaticFiles(directory="static"), name="static")


#cors 코드 추가 (추후 수정)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "nova backend"}


#테이블 테스트를 위한 api
@app.get("/db")
def db():
    try:
        db=SessionLocal()
        tables = db.execute(text("SHOW TABLES")).fetchall()
        return {"db_connected": True, "tables": [t[0] for t in tables]}
    except Exception as e:
        return {"db_connected": False, "error": str(e)}
    
    finally:
        db.close()


#MongoDB 및 모델 서버 준비
@app.on_event("startup")
async def startup_event():
    init_mongo()
    await interp_client.start()


@app.on_event("shutdown")
async def shutdown_event():
    await interp_client.close()