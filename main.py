from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

app = FastAPI(
    title="HAIRGATOR Face Analysis API",
    description="Basic working version",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "HAIRGATOR 서버 실행 중! 🎯"}

@app.get("/test")
def test_server():
    return {
        "message": "HAIRGATOR 서버 테스트! 🎯",
        "test_passed": True,
        "status": "working",
        "version": "1.0 basic"
    }

@app.post("/analyze-face")
async def analyze_face_endpoint(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
        
        # 임시 기본 결과
        return {
            "status": "success",
            "data": {
                "face_shape": "타원형",
                "confidence": 80,
                "analysis_method": "basic_fallback",
                "reasoning": "기본 분석 모드"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
