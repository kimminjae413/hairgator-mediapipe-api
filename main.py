# main.py의 핵심 함수들만 수정 (Perplexity 권장사항 적용)

def extract_measurements_from_20_points(points, width, height):
    """20개 포인트에서 HTML 알고리즘용 측정값 추출 - 디버깅 강화"""
    
    def euclidean_distance_points(p1, p2):
        """두 포인트 간 유클리드 거리 계산"""
        dx = p1['x'] - p2['x']
        dy = p1['y'] - p2['y']
        return math.sqrt(dx * dx + dy * dy)
    
    try:
        # 🔍 Perplexity 권장: 랜드마크 좌표 실제값 로그
        print("🔍 실제 랜드마크 좌표 확인:")
        print(f"  이마 좌: ({points['forehead_left']['x']}, {points['forehead_left']['y']})")
        print(f"  이마 우: ({points['forehead_right']['x']}, {points['forehead_right']['y']})")
        print(f"  광대 좌: ({points['cheekbone_left']['x']}, {points['cheekbone_left']['y']})")
        print(f"  광대 우: ({points['cheekbone_right']['x']}, {points['cheekbone_right']['y']})")
        print(f"  턱 좌: ({points['jaw_left']['x']}, {points['jaw_left']['y']})")
        print(f"  턱 우: ({points['jaw_right']['x']}, {points['jaw_right']['y']})")
        
        # HTML과 동일한 핵심 측정값들
        forehead_width = euclidean_distance_points(
            points['forehead_left'], points['forehead_right']
        )
        
        cheekbone_width = euclidean_distance_points(
            points['cheekbone_left'], points['cheekbone_right']
        )
        
        jaw_width = euclidean_distance_points(
            points['jaw_left'], points['jaw_right']
        )
        
        face_length = euclidean_distance_points(
            points['forehead_top'], points['chin_bottom']
        )
        
        # 정규화 기준: 동공간 거리 (HTML 로직과 동일)
        interpupillary_distance = euclidean_distance_points(
            points['eye_left'], points['eye_right']
        )
        
        # 🔍 Perplexity 권장: 측정값 실제 변화 확인
        print(f"🔍 실제 측정값 (픽셀):")
        print(f"  이마폭: {forehead_width:.1f}px")
        print(f"  광대폭: {cheekbone_width:.1f}px") 
        print(f"  턱폭: {jaw_width:.1f}px")
        print(f"  얼굴길이: {face_length:.1f}px")
        print(f"  동공간거리: {interpupillary_distance:.1f}px")
        
        # 🔍 정규화된 비율값 확인
        norm_forehead = forehead_width / interpupillary_distance
        norm_cheekbone = cheekbone_width / interpupillary_distance
        norm_jaw = jaw_width / interpupillary_distance
        norm_face_length = face_length / interpupillary_distance
        
        print(f"🔍 정규화 비율:")
        print(f"  이마폭 비율: {norm_forehead:.3f}")
        print(f"  광대폭 비율: {norm_cheekbone:.3f}")
        print(f"  턱폭 비율: {norm_jaw:.3f}")
        print(f"  얼굴길이 비율: {norm_face_length:.3f}")
        
        return {
            # HTML 로직: 동공간 거리로 정규화
            "foreheadWidth": norm_forehead,
            "cheekboneWidth": norm_cheekbone,
            "jawWidth": norm_jaw,
            "faceLength": norm_face_length,
            "interpupillaryDistance": interpupillary_distance,
            # 원본 픽셀값 (표시용)
            "foreheadWidthPx": round(forehead_width),
            "cheekboneWidthPx": round(cheekbone_width),
            "jawWidthPx": round(jaw_width),
            "faceLengthPx": round(face_length)
        }
        
    except Exception as e:
        print(f"❌ 측정값 추출 실패: {e}")
        print("🔍 예외 발생으로 fallback 실행됨")
        return generate_safe_measurements(width, height)

def classify_face_shape_scientific_html_logic(measurements):
    """HTML 로직 + Perplexity 권장 디버깅"""
    
    # 🔍 입력값 검증
    print(f"🔍 분류 함수 입력값:")
    print(f"  forehead: {measurements.get('foreheadWidth', 'MISSING')}")
    print(f"  cheekbone: {measurements.get('cheekboneWidth', 'MISSING')}")
    print(f"  jaw: {measurements.get('jawWidth', 'MISSING')}")
    print(f"  face_length: {measurements.get('faceLength', 'MISSING')}")
    
    forehead_width = measurements["foreheadWidth"]
    cheekbone_width = measurements["cheekboneWidth"] 
    jaw_width = measurements["jawWidth"]
    face_length = measurements["faceLength"]
    
    # HTML과 동일한 핵심 비율들
    ratio_FC = face_length / cheekbone_width
    ratio_FW_CW = forehead_width / cheekbone_width
    ratio_CW_JW = cheekbone_width / jaw_width
    
    print(f"🔍 핵심 비율 계산:")
    print(f"  ratio_FC (얼굴길이/광대폭): {ratio_FC:.3f}")
    print(f"  ratio_FW_CW (이마폭/광대폭): {ratio_FW_CW:.3f}")
    print(f"  ratio_CW_JW (광대폭/턱폭): {ratio_CW_JW:.3f}")
    
    face_shape = ""
    confidence = 0
    reasoning = ""
    condition_met = ""
    
    # 🔍 각 조건 체크 과정 로그
    if ratio_FW_CW > 1.07 and forehead_width > cheekbone_width and cheekbone_width > jaw_width:
        face_shape = '하트형'
        confidence = min(95, 75 + (ratio_FW_CW - 1.07) * 100)
        reasoning = f"이마폭/광대폭 비율: {ratio_FW_CW:.3f} > 1.07"
        condition_met = "조건1: 하트형"
        
    elif (cheekbone_width > forehead_width and cheekbone_width > jaw_width and 
          ratio_CW_JW >= 1.10 and ratio_FW_CW < 0.95):
        face_shape = '다이아몬드형'
        confidence = min(93, 73 + (ratio_CW_JW - 1.10) * 150)
        reasoning = f"광대폭이 최대, 광대폭/턱폭: {ratio_CW_JW:.3f}"
        condition_met = "조건2: 다이아몬드형"
        
    elif ratio_FC > 1.5:
        face_shape = '긴형'
        confidence = min(91, 70 + (ratio_FC - 1.5) * 80)
        reasoning = f"얼굴길이/광대폭 비율: {ratio_FC:.3f} > 1.5"
        condition_met = "조건3: 긴형"
        
    elif (ratio_FC >= 1.0 and ratio_FC <= 1.1 and 
          abs(forehead_width - cheekbone_width) < 0.1 * cheekbone_width):
        face_shape = '둥근형'
        confidence = min(89, 78 + (1.1 - ratio_FC) * 100)
        reasoning = f"얼굴길이/광대폭 비율: {ratio_FC:.3f} (1.0-1.1 범위)"
        condition_met = "조건4: 둥근형"
        
    elif (ratio_FC <= 1.15 and abs(forehead_width - cheekbone_width) < 0.15 * cheekbone_width and
          abs(cheekbone_width - jaw_width) < 0.15 * cheekbone_width):
        face_shape = '각진형'
        confidence = min(87, 72 + (1.15 - ratio_FC) * 100)
        reasoning = f"이마≈광대≈턱, 비율: {ratio_FC:.3f} ≤ 1.15"
        condition_met = "조건5: 각진형"
        
    elif ratio_FC >= 1.3 and ratio_FC <= 1.5:
        face_shape = '타원형'
        confidence = min(92, 82 + (1.4 - abs(ratio_FC - 1.4)) * 100)
        reasoning = f"황금 비율: {ratio_FC:.3f} (1.3-1.5 범위)"
        condition_met = "조건6: 타원형 (정상)"
        
    else:
        # 🔍 경계 케이스 - 실제 측정값 기반 정밀 분석
        print("🔍 기본 조건 미충족 - 경계 케이스 정밀 분석")
        print(f"  ratio_FC: {ratio_FC:.3f}")
        print(f"  ratio_FW_CW: {ratio_FW_CW:.3f}")
        print(f"  ratio_CW_JW: {ratio_CW_JW:.3f}")
        
        # 🔧 실제 측정값 기반 정밀 분류 (랜덤 아님!)
        if ratio_FC > 1.2:  # 얼굴이 긴 편
            if ratio_FW_CW > 1.0:  # 이마가 넓은 편
                face_shape = '타원형'
                confidence = 79
                reasoning = f'긴 타원형 (길이비: {ratio_FC:.3f}, 이마비: {ratio_FW_CW:.3f})'
            else:
                face_shape = '긴형'
                confidence = 77
                reasoning = f'긴형 경향 (길이비: {ratio_FC:.3f})'
        elif ratio_FC < 1.2:  # 얼굴이 짧은 편
            if abs(forehead_width - cheekbone_width) < 0.2 * cheekbone_width:
                face_shape = '둥근형'
                confidence = 76
                reasoning = f'둥근형 경향 (길이비: {ratio_FC:.3f}, 폭 유사)'
            else:
                face_shape = '각진형'
                confidence = 74
                reasoning = f'각진형 경향 (길이비: {ratio_FC:.3f})'
        else:  # 중간값
            if ratio_FW_CW > 1.02:  # 이마가 약간 더 넓음
                face_shape = '하트형'
                confidence = 73
                reasoning = f'약한 하트형 (이마비: {ratio_FW_CW:.3f})'
            elif ratio_CW_JW > 1.08:  # 광대가 약간 더 넓음
                face_shape = '다이아몬드형'
                confidence = 71
                reasoning = f'약한 다이아몬드형 (광대비: {ratio_CW_JW:.3f})'
            else:
                face_shape = '타원형'
                confidence = 75
                reasoning = f'표준 타원형 (균형적 비율)'
        
        condition_met = f"경계 케이스: 정밀 분석 → {face_shape}"
    
    print(f"🔍 분류 결과:")
    print(f"  충족 조건: {condition_met}")
    print(f"  최종 얼굴형: {face_shape}")
    print(f"  신뢰도: {confidence}%")
    print(f"  과학적 근거: {reasoning}")
    
    return {
        "faceShape": face_shape,
        "confidence": round(confidence),
        "reasoning": reasoning,
        "ratios": {
            "faceLength_cheekbone": ratio_FC,
            "forehead_cheekbone": ratio_FW_CW,
            "cheekbone_jaw": ratio_CW_JW
        }
    }

def generate_safe_measurements(width, height):
    """안전한 기본 측정값 생성 - 실제 얼굴 비율 기반"""
    print("🔍 generate_safe_measurements 호출됨 (측정 실패시)")
    
    # 🔧 실제 얼굴 비율 기반 추정값 (랜덤 아님!)
    # 이미지 크기 기반으로 현실적인 얼굴 비율 계산
    
    # 기본 얼굴 크기 추정 (이미지 크기 기반)
    estimated_face_width = width * 0.6  # 얼굴이 이미지의 60% 정도
    estimated_face_height = height * 0.8  # 얼굴이 이미지의 80% 정도
    
    # 표준 얼굴 비율 적용 (의학적 기준)
    estimated_forehead = estimated_face_width * 0.85    # 이마폭 = 얼굴폭의 85%
    estimated_cheekbone = estimated_face_width * 0.95   # 광대폭 = 얼굴폭의 95%
    estimated_jaw = estimated_face_width * 0.80         # 턱폭 = 얼굴폭의 80%
    estimated_length = estimated_face_height * 0.75     # 얼굴길이 = 얼굴높이의 75%
    
    # 동공간 거리 표준값 (성인 평균 65px)
    interpupillary = 65
    
    print(f"🔍 이미지 크기 기반 측정값 추정:")
    print(f"  이미지: {width}x{height}")
    print(f"  추정 얼굴크기: {estimated_face_width:.0f}x{estimated_face_height:.0f}")
    print(f"  이마폭: {estimated_forehead:.0f}px")
    print(f"  광대폭: {estimated_cheekbone:.0f}px")
    print(f"  턱폭: {estimated_jaw:.0f}px")
    print(f"  얼굴길이: {estimated_length:.0f}px")
    
    return {
        "foreheadWidth": estimated_forehead / interpupillary,
        "cheekboneWidth": estimated_cheekbone / interpupillary,
        "jawWidth": estimated_jaw / interpupillary,
        "faceLength": estimated_length / interpupillary,
        "interpupillaryDistance": interpupillary,
        "foreheadWidthPx": round(estimated_forehead),
        "cheekboneWidthPx": round(estimated_cheekbone),
        "jawWidthPx": round(estimated_jaw),
        "faceLengthPx": round(estimated_length)
    }
