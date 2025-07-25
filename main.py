# MediaPipe 디버깅을 위한 main.py 패치
# 기존 analyze_with_enhanced_mediapipe 함수를 다음으로 교체

def analyze_with_enhanced_mediapipe(image):
    """HTML 로직 통합 - MediaPipe 고도화 분석 (디버깅 강화)"""
    try:
        height, width = image.shape[:2]
        print(f"🔍 디버그: 이미지 크기 확인 - {width}x{height}")
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"🔍 디버그: RGB 변환 완료")
        
        # MediaPipe 얼굴 메시 감지
        print(f"🔍 디버그: MediaPipe 얼굴 메시 처리 시작")
        results = face_mesh.process(rgb_image)
        print(f"🔍 디버그: MediaPipe 처리 완료")
        
        if results.multi_face_landmarks:
            print(f"🔍 디버그: 얼굴 랜드마크 감지됨 - {len(results.multi_face_landmarks)}개 얼굴")
            landmarks = results.multi_face_landmarks[0]
            print(f"🔍 디버그: 랜드마크 수 - {len(landmarks.landmark)}개")
            
            print("🔬 HTML 통합 로직으로 정밀 분석 시작")
            
            # 🎯 HTML 로직: 정밀 측정값 추출
            try:
                measurements = extract_precise_measurements_html_logic(landmarks, width, height)
                print(f"🔍 디버그: 측정값 추출 성공")
                print(f"  - 이마폭: {measurements.get('foreheadWidthPx', 'N/A')}px")
                print(f"  - 광대폭: {measurements.get('cheekboneWidthPx', 'N/A')}px")
                print(f"  - 턱폭: {measurements.get('jawWidthPx', 'N/A')}px")
                print(f"  - 얼굴길이: {measurements.get('faceLengthPx', 'N/A')}px")
            except Exception as e:
                print(f"❌ 디버그: 측정값 추출 실패 - {e}")
                raise e
            
            # 🎯 HTML 로직: 과학적 얼굴형 분류
            try:
                face_result = classify_face_shape_scientific_html_logic(measurements)
                print(f"🔍 디버그: 분류 완료 - {face_result['faceShape']} ({face_result['confidence']}%)")
                print(f"🔍 디버그: 과학적 근거 - {face_result['reasoning']}")
            except Exception as e:
                print(f"❌ 디버그: 얼굴형 분류 실패 - {e}")
                raise e
            
            # 🎯 220개 상세 랜드마크 추출
            try:
                detailed_coordinates = extract_detailed_landmarks_220(landmarks, width, height)
                print(f"🔍 디버그: 220개 랜드마크 추출 완료 - {len(detailed_coordinates)}개")
            except Exception as e:
                print(f"❌ 디버그: 220개 랜드마크 추출 실패 - {e}")
                raise e
            
            print(f"✅ MediaPipe 고도화 분석 완료: {face_result['faceShape']} ({face_result['confidence']}%)")
            
            return {
                "face_shape": face_result["faceShape"],
                "confidence": face_result["confidence"],
                "coordinates": detailed_coordinates,
                "metrics": measurements,
                "scientific_analysis": {
                    "reasoning": face_result["reasoning"],
                    "ratios": face_result["ratios"],
                    "method": "HTML_integrated_scientific_analysis"
                },
                "landmark_count": len(detailed_coordinates),
                "details": [
                    f"{face_result['faceShape']} (HTML 통합 분석)",
                    f"과학적 근거: {face_result['reasoning']}",
                    f"신뢰도: {face_result['confidence']}%",
                    f"220개 정밀 랜드마크 활용"
                ]
            }
        else:
            # 얼굴 감지 실패
            print("❌ 디버그: MediaPipe 얼굴 감지 실패 - multi_face_landmarks가 None")
            print("⚠️ MediaPipe 얼굴 감지 실패, 고도화된 OpenCV 분석으로 대체")
            return analyze_with_enhanced_opencv(image)
            
    except Exception as e:
        print(f"❌ 디버그: MediaPipe 고도화 분석 전체 오류 - {str(e)}")
        print(f"❌ 디버그: 오류 타입 - {type(e).__name__}")
        import traceback
        print(f"❌ 디버그: 스택 트레이스 - {traceback.format_exc()}")
        return analyze_with_enhanced_opencv(image)

# 추가로 extract_precise_measurements_html_logic 함수도 디버깅 강화
def extract_precise_measurements_html_logic(landmarks, width, height):
    """HTML 로직 통합: 정밀 측정값 추출 (디버깅 강화)"""
    
    def euclidean_distance(p1, p2):
        """유클리드 거리 계산 (HTML과 동일한 로직)"""
        dx = (p1.x - p2.x) * width
        dy = (p1.y - p2.y) * height
        return math.sqrt(dx * dx + dy * dy)
    
    # HTML에서 사용하는 정확한 인덱스들
    try:
        print(f"🔍 디버그: 랜드마크 추출 시작")
        
        # 정규화 기준: 동공간 거리 (HTML 로직과 동일)
        left_eye = landmarks.landmark[33]   # HTML: landmarks[33]
        right_eye = landmarks.landmark[362] # HTML: landmarks[362]
        interpupillary_distance = euclidean_distance(left_eye, right_eye)
        print(f"🔍 디버그: 동공간 거리 - {interpupillary_distance:.1f}px")
        
        # HTML과 동일한 핵심 측정점들
        forehead_left = landmarks.landmark[127]  # HTML: landmarks[127]
        forehead_right = landmarks.landmark[356] # HTML: landmarks[356]
        cheekbone_left = landmarks.landmark[234] # HTML: landmarks[234]
        cheekbone_right = landmarks.landmark[454] # HTML: landmarks[454]
        jaw_left = landmarks.landmark[109]       # HTML: landmarks[109]
        jaw_right = landmarks.landmark[338]      # HTML: landmarks[338]
        face_top = landmarks.landmark[10]        # HTML: landmarks[10]
        face_bottom = landmarks.landmark[152]    # HTML: landmarks[152]
        
        print(f"🔍 디버그: 핵심 랜드마크 좌표 추출 완료")
        
        # HTML과 동일한 측정 방식
        forehead_width = euclidean_distance(forehead_left, forehead_right)
        cheekbone_width = euclidean_distance(cheekbone_left, cheekbone_right)
        jaw_width = euclidean_distance(jaw_left, jaw_right)
        face_length = euclidean_distance(face_top, face_bottom)
        
        print(f"📏 HTML 로직 측정 완료:")
        print(f"  - 동공간 거리: {interpupillary_distance:.1f}px")
        print(f"  - 이마폭: {forehead_width:.1f}px")
        print(f"  - 광대폭: {cheekbone_width:.1f}px")
        print(f"  - 턱폭: {jaw_width:.1f}px")
        print(f"  - 얼굴길이: {face_length:.1f}px")
        
        result = {
            # HTML 로직: 동공간 거리로 정규화
            "foreheadWidth": forehead_width / interpupillary_distance,
            "cheekboneWidth": cheekbone_width / interpupillary_distance,
            "jawWidth": jaw_width / interpupillary_distance,
            "faceLength": face_length / interpupillary_distance,
            "interpupillaryDistance": interpupillary_distance,
            # 원본 픽셀값 (표시용)
            "foreheadWidthPx": round(forehead_width),
            "cheekboneWidthPx": round(cheekbone_width),
            "jawWidthPx": round(jaw_width),
            "faceLengthPx": round(face_length)
        }
        
        print(f"🔍 디버그: 최종 측정값 - {result}")
        return result
        
    except IndexError as e:
        print(f"❌ 디버그: 랜드마크 인덱스 오류 - {e}")
        print(f"❌ 디버그: landmarks.landmark 길이 - {len(landmarks.landmark)}")
        raise e
    except Exception as e:
        print(f"❌ 디버그: 측정값 추출 일반 오류 - {e}")
        raise e
