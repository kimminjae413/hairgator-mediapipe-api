# HAIRGATOR MediaPipe API v7.2

## 🔥 자동 Firebase 파일 감지 시스템 기반 얼굴형 분석 & 헤어스타일 추천 API

FastAPI + MediaPipe + Firebase Storage를 활용한 정밀 얼굴형 분석 및 실시간 헤어스타일 추천 서버

## ✨ 주요 기능

- **🤖 MediaPipe 기반 정밀 얼굴형 분석**: 18개 핵심 랜드마크 기반 해부학적 정확성
- **🎨 퍼스널컬러 분석**: RGB 기반 웜톤/쿨톤 분류
- **🔄 자동 Firebase 파일 감지**: 새 파일 업로드시 자동 인식 (5분 캐싱)
- **📱 실시간 헤어스타일 추천**: 얼굴형별 최적 4개 스타일 추천
- **⚡ 무제한 확장성**: 500개 → 1000개+ 파일 지원

## 🚀 v7.2 새로운 기능

### 자동 Firebase 감지 시스템
- **완전 자동화**: 새 파일 업로드시 코드 수정 없이 자동 감지
- **스마트 캐싱**: 5분마다 자동 갱신, Firebase API 호출 최소화
- **실시간 모니터링**: `/firebase-status` 엔드포인트로 업로드 현황 확인

### 지원하는 얼굴형
- 둥근형, 타원형(계란형), 각진형, 긴형, 하트형, 다이아몬드형

### Firebase Storage 파일 구조
```
hairgator500/
├── 001_클래식보브_둥근형_1020대_v1.jpg.jpg
├── 002_클래식보브_둥근형_1020대_v2.jpg.jpg
├── 003_클래식보브_둥근형_3040대_v1.jpg.jpg
...
├── 500_특수스타일_다이아몬드형_50대이상_v2.jpg.jpg
```

## 📋 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 서버 실행
```bash
python main.py
```

### 3. API 접근
- **기본 URL**: `http://localhost:8000`
- **API 문서**: `http://localhost:8000/docs`

## 🔌 API 엔드포인트

### POST /analyze-face/
얼굴 이미지 업로드하여 얼굴형 분석 및 헤어스타일 추천

**Request**: `multipart/form-data`
- `file`: 이미지 파일 (JPG, PNG)

**Response**:
```json
{
  "status": "success",
  "data": {
    "face_shape": "둥근형",
    "confidence": 90,
    "personal_color": {
      "undertone": "웜톤",
      "confidence": 85,
      "recommended_hair_colors": ["골든브라운", "카라멜브라운"]
    },
    "recommended_hairstyles": [
      {
        "style_name": "볼륨펌",
        "description": "둥근형의 1:1 비율을 1:1.5로 보정...",
        "firebase_urls": ["https://firebasestorage.googleapis.com/..."],
        "primary_image": "https://firebasestorage.googleapis.com/...",
        "total_variations": 2
      }
    ]
  }
}
```

### GET /firebase-status
실시간 Firebase 업로드 상태 및 감지된 파일 현황

### POST /refresh-cache
Firebase 파일 캐시 수동 갱신

### GET /health
서버 상태 및 기능 활성화 확인

## 🛠️ 기술 스택

- **Backend**: FastAPI 0.104.1
- **AI/ML**: MediaPipe 0.10.21, OpenCV, NumPy
- **Storage**: Firebase Storage
- **HTTP Client**: aiohttp 3.9.1
- **Image Processing**: Pillow

## 📊 분석 정확도

- **얼굴형 분류**: 88-94% 신뢰도
- **퍼스널컬러**: 65-85% 신뢰도
- **랜드마크 감지**: MediaPipe 0.7+ 신뢰도

## 🔄 자동 업데이트 시스템

### Firebase 파일 감지 흐름
1. **API 호출시 캐시 확인** (5분 만료)
2. **만료시 Firebase Storage API 호출**
3. **파일 목록 자동 파싱 및 매핑**
4. **새로운 스타일 자동 추가**

### 지원하는 파일명 패턴
- `XXX_스타일명_얼굴형_연령대_변형.jpg.jpg`
- 예: `117_레이어드미디움_타원형_1020대_v1.jpg.jpg`

## 🚀 배포

### Heroku
```bash
git push heroku main
```

### Docker
```bash
docker build -t hairgator-api .
docker run -p 8000:8000 hairgator-api
```

## 📈 확장성

- **현재**: 500개 헤어스타일 지원
- **확장 가능**: 무제한 (자동 감지)
- **새 스타일 추가**: Firebase 업로드만으로 자동 인식

## 🔧 환경 변수

- `PORT`: 서버 포트 (기본값: 8000)

## 📞 지원

문제 발생시 이슈 등록 또는 코드 리뷰 요청

---

**HAIRGATOR v7.2** - 완전 자동화된 AI 헤어스타일 추천 시스템 🎊
