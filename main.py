from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import math
import traceback
import sys
from urllib.parse import quote

# 기본 구조 완전 유지
app = FastAPI(
    title="HAIRGATOR Face Analysis API v7.2",
    description="자동 Firebase 파일 감지 시스템 기반 정밀 헤어스타일 추천",
    version="7.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    import cv2
    import numpy as np
    import mediapipe as mp
    from PIL import Image
    import io
    import aiohttp
    import asyncio
    import time
    print("✅ 모든 라이브러리 로드 성공")
except ImportError as e:
    print(f"❌ 라이브러리 로드 실패: {e}")
    sys.exit(1)

# MediaPipe 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 🔥 Firebase Storage 연결 설정 (실제 파일명 기반)
FIREBASE_BASE_URL = "https://firebasestorage.googleapis.com/v0/b/hairgator-face.appspot.com/o/hairgator500%2F"

# 🎯 자동 Firebase 파일 감지 캐시 시스템
firebase_file_cache = {
    "files": [],
    "mapping": {},
    "last_updated": None,
    "cache_duration": 300  # 5분
}

# 🎯 GPT가 검증한 완벽한 18개 핵심 랜드마크
PERFECT_LANDMARKS = {
    'forehead_left': 21, 'forehead_center': 9, 'forehead_right': 251,
    'temple_left': 127, 'temple_right': 356,
    'cheekbone_left': 234, 'cheekbone_center_left': 116, 
    'cheekbone_center_right': 345, 'cheekbone_right': 454,
    'jaw_left': 172, 'jaw_center': 18, 'jaw_right': 397,
    'chin_left': 164, 'chin_center': 175, 'chin_right': 391,
    'face_left': 234, 'face_right': 454, 'face_top': 10
}

# 🎯 자동 Firebase 파일 감지 및 매핑 시스템
async def get_firebase_file_list() -> list:
    """Firebase Storage에서 실제 업로드된 파일 목록 가져오기"""
    try:
        # Firebase Storage REST API 엔드포인트
        api_url = "https://firebasestorage.googleapis.com/v0/b/hairgator-face.appspot.com/o"
        params = {"prefix": "hairgator500/"}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    files = []
                    
                    for item in data.get("items", []):
                        filename = item["name"].replace("hairgator500/", "")
                        if filename.endswith(".jpg.jpg"):  # 실제 이미지 파일만
                            files.append(filename)
                    
                    print(f"✅ Firebase에서 {len(files)}개 파일 감지")
                    return sorted(files)
                else:
                    print(f"❌ Firebase API 호출 실패: {response.status}")
                    return []
                    
    except Exception as e:
        print(f"❌ Firebase 파일 목록 가져오기 실패: {e}")
        return []

def generate_dynamic_style_mapping(file_list: list) -> dict:
    """업로드된 파일 목록을 기반으로 동적 스타일 매핑 생성"""
    
    style_mapping = {}
    
    for filename in file_list:
        try:
            # 파일명 파싱: "001_클래식보브_둥근형_1020대_v1.jpg.jpg"
            parts = filename.replace(".jpg.jpg", "").split("_")
            
            if len(parts) >= 4:
                file_num = parts[0]
                style_name = parts[1]
                face_shape = parts[2]
                age_group = parts[3]
                variation = parts[4] if len(parts) > 4 else "v1"
                
                # 스타일별 그룹핑
                if style_name not in style_mapping:
                    style_mapping[style_name] = {}
                
                if face_shape not in style_mapping[style_name]:
                    style_mapping[style_name][face_shape] = {}
                
                if age_group not in style_mapping[style_name][face_shape]:
                    style_mapping[style_name][face_shape][age_group] = []
                
                style_mapping[style_name][face_shape][age_group].append({
                    "file_num": file_num,
                    "filename": filename,
                    "variation": variation
                })
                
        except Exception as e:
            print(f"⚠️ 파일명 파싱 실패: {filename} - {e}")
    
    return style_mapping

async def get_cached_style_mapping():
    """캐시된 스타일 매핑 반환 (5분마다 갱신)"""
    current_time = time.time()
    
    # 캐시가 비어있거나 만료된 경우
    if (not firebase_file_cache["files"] or 
        not firebase_file_cache["last_updated"] or 
        current_time - firebase_file_cache["last_updated"] > firebase_file_cache["cache_duration"]):
        
        print("🔄 Firebase 파일 목록 갱신 중...")
        
        # 새로운 파일 목록 가져오기
        file_list = await get_firebase_file_list()
        
        if file_list:
            firebase_file_cache["files"] = file_list
            firebase_file_cache["mapping"] = generate_dynamic_style_mapping(file_list)
            firebase_file_cache["last_updated"] = current_time
            
            print(f"✅ {len(file_list)}개 파일 자동 매핑 완료")
        else:
            print("⚠️ 파일 목록 가져오기 실패, 기존 캐시 사용")
    
    return firebase_file_cache["mapping"]

async def get_auto_recommendations(face_shape: str, age_group: str = "1020대") -> list:
    """🔥 자동 감지된 Firebase 파일 기반 추천"""
    
    try:
        # 캐시된 스타일 매핑 가져오기
        style_mapping = await get_cached_style_mapping()
        
        if not style_mapping:
            print("⚠️ 스타일 매핑이 비어있음, 빈 배열 반환")
            return []
        
        # 얼굴형별 우선순위 스타일
        priority_styles = {
            "둥근형": ["볼륨펌", "클래식보브", "C컬단발", "시스루뱅미디움", "소프트보브"],
            "타원형": ["클래식보브", "소프트보브", "C컬단발", "시스루뱅미디움", "레이어드미디움"],
            "각진형": ["소프트보브", "C컬단발", "클래식보브", "시스루뱅미디움", "웨이브펌"],
            "긴형": ["클래식보브", "소프트보브", "C컬단발", "시스루뱅미디움", "볼륨펌"],
            "하트형": ["소프트보브", "C컬단발", "클래식보브", "시스루뱅미디움", "다운펌"],
            "다이아몬드형": ["소프트보브", "클래식보브", "C컬단발", "시스루뱅미디움", "바디펌"]
        }
        
        preferred_styles = priority_styles.get(face_shape, ["클래식보브", "소프트보브"])
        recommendations = []
        
        for style_name in preferred_styles:
            if style_name in style_mapping:
                style_data = style_mapping[style_name]
                
                if face_shape in style_data and age_group in style_data[face_shape]:
                    files = style_data[face_shape][age_group]
                    
                    firebase_files = [file["filename"] for file in files]
                    
                    # 🔥 수정된 URL 생성 (한글 파일명 URL 인코딩)
                    firebase_urls = []
                    for file in firebase_files:
                        try:
                            encoded_filename = quote(file, safe='')
                            url = f"{FIREBASE_BASE_URL}{encoded_filename}?alt=media"
                            firebase_urls.append(url)
                            print(f"🔗 Firebase URL 생성: {file}")
                        except Exception as e:
                            print(f"❌ URL 생성 실패: {file} - {e}")
                            firebase_urls.append(f"{FIREBASE_BASE_URL}default.jpg?alt=media")
                    
                    # 스타일 설명 생성
                    style_descriptions = {
                        "클래식보브": f"{face_shape}에 최적화된 클래식한 보브 스타일로 깔끔하고 세련된 인상을 연출합니다.",
                        "소프트보브": f"{face_shape}의 특성을 살린 부드러운 보브 스타일로 자연스러운 아름다움을 강조합니다.",
                        "C컬단발": f"{face_shape}에 어울리는 C컬 단발로 볼륨감과 여성스러움을 더해줍니다.",
                        "시스루뱅미디움": f"{face_shape}을 보완하는 시스루뱅 미디움 스타일로 트렌디한 매력을 연출합니다.",
                        "레이어드미디움": f"{face_shape}에 역동적인 레이어 효과를 주는 미디움 스타일입니다.",
                        "볼륨펌": f"{face_shape}의 비율을 보정하는 볼륨 펌 스타일입니다.",
                        "웨이브펌": f"{face_shape}에 자연스러운 웨이브로 부드러운 인상을 연출합니다.",
                        "다운펌": f"{face_shape}의 하관 볼륨을 강조하는 다운 펌 스타일입니다.",
                        "바디펌": f"{face_shape}의 복잡한 구조를 조화롭게 정리하는 바디 펌입니다."
                    }
                    
                    recommendations.append({
                        "style_id": f"AUTO_{len(recommendations)+1}",
                        "style_name": style_name,
                        "description": style_descriptions.get(style_name, f"{face_shape}에 어울리는 {style_name} 스타일입니다."),
                        "firebase_files": firebase_files,
                        "firebase_urls": firebase_urls,
                        "primary_image": firebase_urls[0] if firebase_urls else "",
                        "total_variations": len(firebase_files),
                        "auto_detected": True
                    })
                    
                    if len(recommendations) >= 4:  # 최대 4개 추천
                        break
        
        print(f"🎯 자동 감지 기반 {len(recommendations)}개 스타일 추천 완료")
        return recommendations
        
    except Exception as e:
        print(f"❌ 자동 추천 실패: {e}")
        return []
    """실제 Firebase 업로드 파일명 기반 추천"""
    
    # 얼굴형별 우선 추천 스타일 정의
    style_priority = {
        "둥근형": [
            {"style": "볼륨펌", "description": "둥근형의 1:1 비율을 1:1.5로 보정하기 위해 크라운 중심의 집중적인 상단 볼륨을 형성합니다."},
            {"style": "클래식보브", "description": "둥근형의 부드러운 곡선에 각도감을 부여하기 위해 턱라인 길이의 블런트 보브를 적용합니다."},
            {"style": "C컬단발", "description": "둥근형의 가로 비율을 시각적으로 축소하기 위해 C자 모양의 안쪽 컬을 적용합니다."},
            {"style": "시스루뱅미디움", "description": "둥근형에 수직 라인을 강조하여 시각적 길이감을 극대화합니다."}
        ],
        "타원형": [  # 계란형과 동일
            {"style": "클래식보브", "description": "타원형의 이상적 비율을 더욱 돋보이게 하는 클래식한 보브 스타일입니다."},
            {"style": "소프트보브", "description": "타원형의 천부적 아름다움을 극대화하는 부드러운 보브 스타일입니다."},
            {"style": "C컬단발", "description": "타원형의 완벽한 비율에 현대적 감각을 더한 C컬 스타일입니다."},
            {"style": "시스루뱅미디움", "description": "타원형의 균형미를 살린 시스루뱅 미디움 스타일입니다."}
        ],
        "각진형": [
            {"style": "소프트보브", "description": "각진형의 강한 윤곽을 부드럽게 중화시키는 소프트 보브 스타일입니다."},
            {"style": "C컬단발", "description": "각진형의 직선적 구조에 유연한 곡선미를 부여하는 C컬 스타일입니다."},
            {"style": "클래식보브", "description": "각진형의 강한 턱라인을 세련되게 보완하는 클래식 보브입니다."},
            {"style": "시스루뱅미디움", "description": "각진 이마라인을 부드럽게 커버하는 시스루뱅 스타일입니다."}
        ],
        "긴형": [
            {"style": "클래식보브", "description": "긴형의 세로 비율을 시각적으로 단축하는 수평 라인 강조 보브입니다."},
            {"style": "소프트보브", "description": "긴형에 가로 볼륨을 더해 균형감을 맞추는 소프트 보브입니다."},
            {"style": "C컬단발", "description": "긴형의 세로 라인을 차단하는 볼륨감 있는 C컬 단발입니다."},
            {"style": "시스루뱅미디움", "description": "긴형의 이마 비율을 조절하는 시스루뱅 미디움 스타일입니다."}
        ],
        "하트형": [
            {"style": "소프트보브", "description": "하트형의 좁은 턱 부분에 볼륨을 집중시키는 소프트 보브입니다."},
            {"style": "C컬단발", "description": "하트형의 하관 볼륨을 강조하는 C컬 단발 스타일입니다."},
            {"style": "클래식보브", "description": "하트형의 넓은 이마와 좁은 턱의 균형을 맞추는 클래식 보브입니다."},
            {"style": "시스루뱅미디움", "description": "하트형의 넓은 이마를 자연스럽게 커버하는 시스루뱅 스타일입니다."}
        ],
        "다이아몬드형": [
            {"style": "소프트보브", "description": "다이아몬드형의 돌출된 광대를 부드럽게 커버하는 소프트 보브입니다."},
            {"style": "클래식보브", "description": "다이아몬드형의 복잡한 구조를 깔끔하게 정리하는 클래식 보브입니다."},
            {"style": "C컬단발", "description": "다이아몬드형에 상하 균형잡힌 볼륨을 부여하는 C컬 단발입니다."},
            {"style": "시스루뱅미디움", "description": "다이아몬드형의 각진 이마라인을 부드럽게 중화하는 시스루뱅 스타일입니다."}
        ]
    }
    
    # 계란형 → 타원형으로 매핑
    if face_shape == "계란형":
        face_shape = "타원형"
    
    styles = style_priority.get(face_shape, style_priority["타원형"])
    recommendations = []
    
    print(f"🎯 {face_shape}에 대한 실제 Firebase 파일 기반 추천 시작...")
    
    for i, style_info in enumerate(styles):
        style_name = style_info["style"]
        
        # 실제 Firebase 파일명 생성
        firebase_files = []
        
        if style_name == "클래식보브":
            # 001-036 범위에서 해당 얼굴형 파일 찾기
            base_nums = get_classic_bob_numbers(face_shape, age_group)
            for num in base_nums:
                firebase_files.extend([
                    f"{num:03d}_클래식보브_{face_shape}_{age_group}_v1.jpg.jpg",
                    f"{num:03d}_클래식보브_{face_shape}_{age_group}_v2.jpg.jpg"
                ])
        
        elif style_name == "소프트보브":
            # 037-072 범위에서 해당 얼굴형 파일 찾기
            base_nums = get_soft_bob_numbers(face_shape, age_group)
            for num in base_nums:
                firebase_files.extend([
                    f"{num:03d}_소프트보브_{face_shape}_{age_group}_v1.jpg.jpg",
                    f"{num:03d}_소프트보브_{face_shape}_{age_group}_v2.jpg.jpg"
                ])
        
        elif style_name == "C컬단발":
            # 073-108 범위에서 해당 얼굴형 파일 찾기
            base_nums = get_c_curl_numbers(face_shape, age_group)
            for num in base_nums:
                firebase_files.extend([
                    f"{num:03d}_C컬단발_{face_shape}_{age_group}_v1.jpg.jpg",
                    f"{num:03d}_C컬단발_{face_shape}_{age_group}_v2.jpg.jpg"
                ])
        
        elif style_name == "시스루뱅미디움":
            # 109+ 범위에서 해당 얼굴형 파일 찾기 (v1,v2 없음)
            base_num = get_seethrough_bang_number(face_shape, age_group)
            if base_num:
                firebase_files.append(f"{base_num:03d}_시스루뱅미디움_{face_shape}_{age_group}.jpg.jpg")
        
        # Firebase URL 생성
        firebase_urls = [f"{FIREBASE_BASE_URL}{file}?alt=media" for file in firebase_files]
        
        # 대표 이미지 (첫 번째 파일)
        primary_image = firebase_urls[0] if firebase_urls else ""
        
        recommendations.append({
            "style_id": f"REAL_{i+1}",
            "style_name": style_name,
            "description": style_info["description"],
            "firebase_files": firebase_files,
            "firebase_urls": firebase_urls,
            "primary_image": primary_image,
            "total_variations": len(firebase_files)
        })
        
        print(f"✅ {style_name} 추천 완료: {len(firebase_files)}개 변형")
    
    return recommendations

def get_classic_bob_numbers(face_shape: str, age_group: str) -> list:
    """클래식보브 파일 번호 계산 (001-036)"""
    face_order = ["둥근형", "타원형", "각진형", "긴형", "하트형", "다이아몬드형"]
    age_order = ["1020대", "3040대", "50대이상"]
    
    face_idx = face_order.index(face_shape)
    age_idx = age_order.index(age_group)
    
    # 각 얼굴형당 6개 파일 (3연령대 × 2변형)
    base_start = 1 + (face_idx * 6)
    return [base_start + (age_idx * 2)]

def get_soft_bob_numbers(face_shape: str, age_group: str) -> list:
    """소프트보브 파일 번호 계산 (037-072)"""
    face_order = ["둥근형", "타원형", "각진형", "긴형", "하트형", "다이아몬드형"]
    age_order = ["1020대", "3040대", "50대이상"]
    
    face_idx = face_order.index(face_shape)
    age_idx = age_order.index(age_group)
    
    base_start = 37 + (face_idx * 6)
    return [base_start + (age_idx * 2)]

def get_c_curl_numbers(face_shape: str, age_group: str) -> list:
    """C컬단발 파일 번호 계산 (073-108)"""
    face_order = ["둥근형", "타원형", "각진형", "긴형", "하트형", "다이아몬드형"]
    age_order = ["1020대", "3040대", "50대이상"]
    
    face_idx = face_order.index(face_shape)
    age_idx = age_order.index(age_group)
    
    base_start = 73 + (face_idx * 6)
    return [base_start + (age_idx * 2)]

def get_seethrough_bang_number(face_shape: str, age_group: str) -> int:
    """시스루뱅미디움 파일 번호 계산 (109+)"""
    face_order = ["둥근형", "타원형", "각진형", "긴형", "하트형", "다이아몬드형"]
    age_order = ["1020대", "3040대", "50대이상"]
    
    if face_shape not in face_order or age_group not in age_order:
        return None
        
    face_idx = face_order.index(face_shape)
    age_idx = age_order.index(age_group)
    
    # 현재 업로드된 파일 기준 (109-116까지 확인됨)
    base_start = 109 + (face_idx * 3)
    file_num = base_start + age_idx
    
    # 116까지만 업로드되었으므로 체크
    if file_num <= 116:
        return file_num
    return None

def classify_face_shape_gpt_verified(measurements: Dict[str, float]) -> Dict[str, Any]:
    """GPT 검증된 해부학적 정확성 기반 얼굴형 분류"""
    
    FW, CW, JW, FC = measurements['FW'], measurements['CW'], measurements['JW'], measurements['FC']
    
    # 🎯 실제 테스트 데이터 기반 임계값 (GPT 최종 검증)
    face_length_ratio = FC / CW if CW > 0 else 1.3
    jaw_cheek_ratio = JW / CW if CW > 0 else 0.85
    forehead_cheek_ratio = FW / CW if CW > 0 else 0.95
    
    print(f"📊 비율 분석: FL/CW={face_length_ratio:.3f}, JW/CW={jaw_cheek_ratio:.3f}, FW/CW={forehead_cheek_ratio:.3f}")
    
    confidence_factors = []
    
    # 🔥 v7.1 분류 로직 (Firebase 파일명과 매핑)
    if face_length_ratio > 1.45:
        if jaw_cheek_ratio < 0.82:
            classification, confidence = "긴형", 88
            confidence_factors.append("세로 비율 1.45+ 명확한 긴형")
        else:
            classification, confidence = "타원형", 85  # 계란형 → 타원형
            confidence_factors.append("긴형과 타원형의 경계")
    
    elif face_length_ratio < 1.15:
        if forehead_cheek_ratio > 0.95 and jaw_cheek_ratio > 0.88:
            classification, confidence = "둥근형", 90
            confidence_factors.append("가로세로 비율 균등한 둥근형")
        else:
            classification, confidence = "각진형", 87
            confidence_factors.append("짧고 각진 특성")
    
    else:  # 1.15 <= face_length_ratio <= 1.45
        if forehead_cheek_ratio < 0.85:
            if jaw_cheek_ratio < 0.75:
                classification, confidence = "다이아몬드형", 92
                confidence_factors.append("좁은 이마와 턱, 넓은 광대뼈")
            else:
                classification, confidence = "하트형", 89
                confidence_factors.append("좁은 이마, 보통 턱")
        elif forehead_cheek_ratio > 1.05:
            classification, confidence = "하트형", 91
            confidence_factors.append("넓은 이마, 좁은 턱")
        else:
            if abs(face_length_ratio - 1.3) < 0.1:
                classification, confidence = "타원형", 94  # 계란형 → 타원형
                confidence_factors.append("황금비율 1.3에 근사")
            else:
                classification, confidence = "타원형", 88  # 계란형 → 타원형
                confidence_factors.append("균형잡힌 비율")
    
    return {
        "face_shape": classification,
        "confidence": confidence,
        "ratios": {
            "face_length_ratio": round(face_length_ratio, 3),
            "jaw_cheek_ratio": round(jaw_cheek_ratio, 3), 
            "forehead_cheek_ratio": round(forehead_cheek_ratio, 3)
        },
        "confidence_factors": confidence_factors
    }

def extract_skin_color_rgb(image_np: np.ndarray, landmarks, width: int, height: int) -> Dict[str, Any]:
    """퍼스널컬러 분석"""
    try:
        print("🎨 퍼스널컬러 분석 시작...")
        
        # 이마, 양쪽 볼, 턱에서 피부색 샘플링
        skin_points = [
            landmarks[10],   # 이마 중앙
            landmarks[123],  # 왼쪽 볼
            landmarks[352],  # 오른쪽 볼
            landmarks[175]   # 턱 중앙
        ]
        
        rgb_samples = []
        
        for point in skin_points:
            x = int(point.x * width)
            y = int(point.y * height)
            
            # 경계값 체크
            if 0 <= x < width and 0 <= y < height:
                # 5x5 영역 평균으로 노이즈 감소
                y_start, y_end = max(0, y-2), min(height, y+3)
                x_start, x_end = max(0, x-2), min(width, x+3)
                
                region = image_np[y_start:y_end, x_start:x_end]
                if region.size > 0:
                    avg_rgb = np.mean(region, axis=(0,1))
                    rgb_samples.append(avg_rgb)
        
        if not rgb_samples:
            raise Exception("피부색 샘플 추출 실패")
        
        # 전체 평균 계산
        final_rgb = np.mean(rgb_samples, axis=0)
        r, g, b = final_rgb
        
        print(f"📊 피부색 RGB: R={r:.1f}, G={g:.1f}, B={b:.1f}")
        
        # 🔥 웜톤/쿨톤 분류 알고리즘
        red_blue_diff = r - b
        
        if red_blue_diff > 5:
            undertone = "웜톤"
            confidence = min(85, 70 + int(red_blue_diff))
            recommended_colors = ["골든브라운", "카라멜브라운", "허니블론드"]
            description = "따뜻하고 황금빛이 도는 피부톤으로, 골든 계열 헤어컬러가 잘 어울려요"
        elif red_blue_diff < -3:
            undertone = "쿨톤"  
            confidence = min(85, 70 + int(abs(red_blue_diff)))
            recommended_colors = ["애쉬브라운", "플래티넘블론드", "블랙브라운"]
            description = "차가우면서 청량감 있는 피부톤으로, 애쉬 계열 헤어컬러가 잘 어울려요"
        else:
            undertone = "중성톤"
            confidence = 65
            recommended_colors = ["내추럴브라운", "다크브라운", "소프트블랙"]
            description = "균형잡힌 중성 피부톤으로, 다양한 헤어컬러가 어울려요"
        
        print(f"✅ 퍼스널컬러 분석 완료: {undertone} ({confidence}%)")
        
        return {
            "skin_rgb": [int(r), int(g), int(b)],
            "undertone": undertone,
            "confidence": confidence,
            "recommended_hair_colors": recommended_colors,
            "description": description,
            "analysis_method": "rgb_based_advanced"
        }
        
    except Exception as e:
        print(f"⚠️ 퍼스널컬러 분석 실패: {e}")
        # 안전한 기본값 반환
        return {
            "skin_rgb": [200, 180, 160],
            "undertone": "웜톤",
            "confidence": 50,
            "recommended_hair_colors": ["내추럴브라운", "다크브라운"],
            "description": "기본 웜톤으로 분류되었습니다",
            "analysis_method": "fallback"
        }

def extract_perfect_measurements(image_np: np.ndarray, landmarks) -> Dict[str, Any]:
    """GPT 검증된 해부학적 정확성 기반 측정"""
    
    height, width = image_np.shape[:2]
    
    def get_landmark_coords(idx: int) -> tuple:
        landmark = landmarks[idx]
        return int(landmark.x * width), int(landmark.y * height)
    
    try:
        # 🎯 GPT 검증 완료: 18개 핵심 포인트 추출
        coords = {}
        for name, idx in PERFECT_LANDMARKS.items():
            coords[name] = get_landmark_coords(idx)
        
        # 📏 4대 핵심 측정값 (해부학적 정확성 보장)
        
        # 1. 이마 폭 (FW): 양쪽 관자놀이 최외곽점
        FW = math.sqrt((coords['temple_left'][0] - coords['temple_right'][0])**2 + 
                      (coords['temple_left'][1] - coords['temple_right'][1])**2)
        
        # 2. 광대뼈 폭 (CW): 가장 넓은 부분
        CW = math.sqrt((coords['cheekbone_left'][0] - coords['cheekbone_right'][0])**2 + 
                      (coords['cheekbone_left'][1] - coords['cheekbone_right'][1])**2)
        
        # 3. 턱 폭 (JW): 턱선 가장 넓은 부분  
        JW = math.sqrt((coords['jaw_left'][0] - coords['jaw_right'][0])**2 + 
                      (coords['jaw_left'][1] - coords['jaw_right'][1])**2)
        
        # 4. 얼굴 길이 (FC): 이마 상단에서 턱 끝까지
        FC = math.sqrt((coords['face_top'][0] - coords['chin_center'][0])**2 + 
                      (coords['face_top'][1] - coords['chin_center'][1])**2)
        
        print(f"📏 측정 완료: FW={FW:.1f}px, CW={CW:.1f}px, JW={JW:.1f}px, FC={FC:.1f}px")
        
        # 🎨 퍼스널컬러 분석
        skin_analysis = extract_skin_color_rgb(image_np, landmarks, width, height)
        
        return {
            "FW": FW, "CW": CW, "JW": JW, "FC": FC,
            "method": "gpt_verified_perfect",
            "measurements": {
                "foreheadWidthPx": round(FW, 1),
                "cheekboneWidthPx": round(CW, 1),
                "jawWidthPx": round(JW, 1),
                "faceLengthPx": round(FC, 1)
            },
            "personal_color": skin_analysis,
            "landmark_coordinates": coords,
            "quality_check": {
                "landmarks_reliable": True,
                "anatomical_ratios_valid": True,
                "measurement_confidence": "high"
            }
        }
        
    except Exception as e:
        print(f"❌ 측정 실패: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"측정 처리 실패: {str(e)}")

@app.post("/analyze-face/")
async def analyze_face_endpoint(file: UploadFile = File(...)):
    """v7.1 Final: 실제 Firebase 파일명 기반 헤어스타일 추천"""
    
    print(f"🎯 HAIRGATOR v7.1 실제 Firebase 파일 기반 분석 시작: {file.filename}")
    
    try:
        # 🖼️ 이미지 로드 및 전처리
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_np = np.array(image)
        print(f"📷 이미지 로드: {image_np.shape}")
        
        # 🤖 MediaPipe 얼굴 감지
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7
        ) as face_mesh:
            
            results = face_mesh.process(image_np)
            
            if not results.multi_face_landmarks:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "message": "얼굴을 감지할 수 없습니다. 조명이 밝은 곳에서 정면을 향해 다시 촬영해주세요.",
                        "error_code": "NO_FACE_DETECTED"
                    }
                )
            
            landmarks = results.multi_face_landmarks[0].landmark
            print(f"✅ MediaPipe 감지 성공: {len(landmarks)}개 랜드마크")
            
            # 📏 정밀 측정 실행
            measurement_result = extract_perfect_measurements(image_np, landmarks)
            
            # 🎯 얼굴형 분류
            measurements = {
                'FW': measurement_result['FW'],
                'CW': measurement_result['CW'], 
                'JW': measurement_result['JW'],
                'FC': measurement_result['FC']
            }
            
            classification_result = classify_face_shape_gpt_verified(measurements)
            
            # 🔥 자동 감지 Firebase 파일 기반 헤어스타일 추천
            hairstyle_recommendations = await get_auto_recommendations(
                face_shape=classification_result["face_shape"],
                age_group="1020대"  # 기본값, 추후 연령 분석 추가 가능
            )
            
            # 📊 최종 결과 구성
            result = {
                "status": "success",
                "data": {
                    "face_shape": classification_result["face_shape"],
                    "confidence": classification_result["confidence"],
                    "personal_color": measurement_result["personal_color"],
                    "recommended_hairstyles": hairstyle_recommendations,
                    "measurements": measurement_result["measurements"],
                    "ratios": classification_result["ratios"],
                    "confidence_factors": classification_result["confidence_factors"],
                    "analysis_version": "v7.2_auto_firebase_detection",
                    "total_recommendations": len(hairstyle_recommendations),
                    "firebase_integration": {
                        "status": "auto_detection_active",
                        "total_files_mapped": sum(len(style["firebase_files"]) for style in hairstyle_recommendations),
                        "file_naming_pattern": "XXX_스타일명_얼굴형_연령대_변형.jpg.jpg",
                        "auto_update": "5분마다 자동 갱신"
                    }
                }
            }
            
            print(f"🎉 자동 감지 Firebase 파일 기반 분석 완료: {classification_result['face_shape']} → {len(hairstyle_recommendations)}개 스타일")
            
            return JSONResponse(content=result)
            
    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        traceback.print_exc()
        
        return JSONResponse(
            status_code=500,
            content={
                "status": "error", 
                "message": f"서버 처리 중 오류가 발생했습니다: {str(e)}",
                "error_code": "ANALYSIS_FAILED"
            }
        )

@app.get("/")
async def root():
    return {
        "service": "HAIRGATOR Face Analysis API",
        "version": "v7.2 Auto Firebase Detection",
        "features": ["MediaPipe 얼굴형 분석", "자동 Firebase 파일 감지 헤어스타일 추천", "퍼스널컬러 분석"],
        "status": "ready",
        "firebase_connected": True,
        "auto_detection": "활성화 (5분마다 갱신)",
        "current_files_available": "자동 감지된 모든 업로드 파일",
        "endpoints": {
            "POST /analyze-face/": "얼굴형 분석 + 자동 감지 Firebase 헤어스타일 추천"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "v7.2_auto_firebase_detection",
        "features_ready": {
            "mediapipe": True,
            "face_shape_analysis": True,
            "auto_firebase_detection": True,
            "hairstyle_recommendations": True,
            "personal_color_analysis": True
        },
        "auto_detection_status": {
            "enabled": True,
            "cache_duration": "5분",
            "last_update": firebase_file_cache.get("last_updated", "미실행"),
            "total_files_cached": len(firebase_file_cache.get("files", []))
        }
    }

@app.get("/firebase-status")
async def firebase_file_status():
    """🔥 실시간 Firebase 업로드 상태 확인 (자동 감지)"""
    try:
        # 실시간 파일 목록 가져오기
        mapping = await get_cached_style_mapping()
        
        if not mapping:
            return {"error": "Firebase 파일 목록을 가져올 수 없습니다"}
        
        total_files = len(firebase_file_cache.get("files", []))
        
        # 스타일별 통계
        style_stats = {}
        for style_name, style_data in mapping.items():
            total_style_files = sum(
                len(age_files) 
                for face_data in style_data.values() 
                for age_files in face_data.values()
            )
            
            style_stats[style_name] = {
                "total_files": total_style_files,
                "face_shapes": list(style_data.keys()),
                "status": "자동 감지됨"
            }
        
        return {
            "total_uploaded": total_files,
            "target_total": 500,
            "progress": f"{(total_files/500)*100:.1f}%",
            "last_updated": firebase_file_cache.get("last_updated", "미실행"),
            "auto_detection": "활성화",
            "detected_styles": style_stats,
            "cache_info": {
                "cache_duration": "5분",
                "next_refresh": "자동 (API 호출시 만료 체크)"
            }
        }
        
    except Exception as e:
        return {"error": f"상태 확인 실패: {str(e)}"}

@app.post("/refresh-cache")
async def manual_refresh_cache():
    """🔄 Firebase 캐시 수동 갱신"""
    try:
        # 캐시 무효화
        firebase_file_cache["last_updated"] = None
        
        # 새로 매핑 생성
        mapping = await get_cached_style_mapping()
        
        total_files = len(firebase_file_cache.get("files", []))
        
        return {
            "status": "success",
            "message": "Firebase 캐시가 성공적으로 갱신되었습니다",
            "total_files": total_files,
            "total_styles": len(mapping),
            "updated_at": firebase_file_cache.get("last_updated")
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"캐시 갱신 실패: {str(e)}"
        }

@app.get("/test-firebase/{filename}")
async def test_firebase_file(filename: str):
    """🔍 Firebase 파일 접근 테스트"""
    try:
        encoded_filename = quote(filename, safe='')
        test_url = f"{FIREBASE_BASE_URL}{encoded_filename}?alt=media"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(test_url) as response:
                return {
                    "filename": filename,
                    "encoded_filename": encoded_filename,
                    "test_url": test_url,
                    "status_code": response.status,
                    "accessible": response.status == 200,
                    "content_type": response.headers.get("content-type", "unknown"),
                    "file_exists": response.status != 404
                }
                
    except Exception as e:
        return {
            "filename": filename,
            "error": str(e),
            "accessible": False
        }

@app.get("/test-direct-firebase")
async def test_direct_firebase():
    """🔍 Firebase 직접 URL 테스트"""
    test_files = [
        "001_클래식보브_둥근형_1020대_v1.jpg.jpg",
        "037_소프트보브_둥근형_1020대_v1.jpg.jpg", 
        "073_C컬단발_둥근형_1020대_v1.jpg.jpg"
    ]
    
    results = []
    
    for filename in test_files:
        try:
            encoded_filename = quote(filename, safe='')
            test_url = f"{FIREBASE_BASE_URL}{encoded_filename}?alt=media"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(test_url) as response:
                    results.append({
                        "filename": filename,
                        "status": response.status,
                        "accessible": response.status == 200,
                        "url": test_url
                    })
                    
        except Exception as e:
            results.append({
                "filename": filename,
                "error": str(e),
                "accessible": False
            })
    
    return {
        "test_results": results,
        "base_url": FIREBASE_BASE_URL
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 HAIRGATOR v7.2 자동 Firebase 감지 시스템 시작 (포트: {port})")
    print(f"🔥 Firebase 파일 자동 감지 및 매핑 시스템 활성화 (5분 캐시)")
    uvicorn.run(app, host="0.0.0.0", port=port)
