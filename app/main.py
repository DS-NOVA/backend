from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db.database import SessionLocal
from sqlalchemy import text
from app.routers import user_router
from app.routers import upload_router
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.include_router(user_router.router)
app.include_router(upload_router.router)
app.mount("/static", StaticFiles(directory="uploads"), name="static") #검출 모델

#cors 코드 추가 (추후 수정)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
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
