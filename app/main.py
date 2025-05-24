from fastapi import FastAPI
from app.db.database import SessionLocal
from sqlalchemy import text
from app.routers import user_router

app = FastAPI()

app.include_router(user_router.router)

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
