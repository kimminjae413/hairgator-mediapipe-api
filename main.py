def classify_face_shape_scientific_html_logic(measurements):
    """HTML 논문 기반 얼굴형 분류 로직 - 최소한 안전 조정"""
    
    forehead_width = measurements["foreheadWidth"]
    cheekbone_width = measurements["cheekboneWidth"] 
    jaw_width = measurements["jawWidth"]
    face_length = measurements["faceLength"]
    
    # HTML과 동일한 핵심 비율들
    ratio_FC = face_length / cheekbone_width
    ratio_FW_CW = forehead_width / cheekbone_width
    ratio_CW_JW = cheekbone_width / jaw_width
    
    print(f"🧮 비율 계산: FC={ratio_FC:.3f}, FW_CW={ratio_FW_CW:.3f}, CW_JW={ratio_CW_JW:.3f}")
    
    # HTML 논문 알고리즘과 거의 동일 - 최소 조정만
    if ratio_FW_CW > 1.05 and forehead_width > cheekbone_width and cheekbone_width > jaw_width:  # 1.07→1.05만
        face_shape = '하트형'
        confidence = min(95, 75 + (ratio_FW_CW - 1.05) * 100)
        reasoning = f"이마폭/광대폭 비율: {ratio_FW_CW:.3f} > 1.05"
        
    elif (cheekbone_width > forehead_width and cheekbone_width > jaw_width and 
          ratio_CW_JW >= 1.10 and ratio_FW_CW < 0.95):  # 기존 그대로
        face_shape = '다이아몬드형'
        confidence = min(93, 73 + (ratio_CW_JW - 1.10) * 150)
        reasoning = f"광대폭이 최대, 광대폭/턱폭: {ratio_CW_JW:.3f}"
        
    elif ratio_FC > 1.45:  # 1.5→1.45만 살짝
        face_shape = '긴형'
        confidence = min(91, 70 + (ratio_FC - 1.45) * 80)
        reasoning = f"얼굴길이/광대폭 비율: {ratio_FC:.3f} > 1.45"
        
    elif (ratio_FC >= 1.0 and ratio_FC <= 1.1 and 
          abs(forehead_width - cheekbone_width) < 0.1 * cheekbone_width):  # 기존 그대로
        face_shape = '둥근형'
        confidence = min(89, 78 + (1.1 - ratio_FC) * 100)
        reasoning = f"얼굴길이/광대폭 비율: {ratio_FC:.3f} (1.0-1.1 범위)"
        
    elif (ratio_FC <= 1.15 and abs(forehead_width - cheekbone_width) < 0.15 * cheekbone_width and
          abs(cheekbone_width - jaw_width) < 0.15 * cheekbone_width):  # 기존 그대로
        face_shape = '각진형'
        confidence = min(87, 72 + (1.15 - ratio_FC) * 100)
        reasoning = f"이마≈광대≈턱, 비율: {ratio_FC:.3f} ≤ 1.15"
        
    elif ratio_FC >= 1.3 and ratio_FC <= 1.5:  # 기존 그대로
        face_shape = '타원형'
        confidence = min(92, 82 + (1.4 - abs(ratio_FC - 1.4)) * 100)
        reasoning = f"황금 비율: {ratio_FC:.3f} (1.3-1.5 범위)"
        
    else:
        # 경계 케이스 - 기존과 거의 동일
        if ratio_FC > 1.2:  # 얼굴이 긴 편
            if ratio_FW_CW > 1.0:  # 이마가 넓은 편
                face_shape = '타원형'
                confidence = 79
                reasoning = f'긴 타원형 (길이비: {ratio_FC:.3f})'
            else:
                face_shape = '긴형'
                confidence = 77
                reasoning = f'긴형 경향 (길이비: {ratio_FC:.3f})'
        elif ratio_FC < 1.2:  # 얼굴이 짧은 편
            if abs(forehead_width - cheekbone_width) < 0.2 * cheekbone_width:
                face_shape = '둥근형'
                confidence = 76
                reasoning = f'둥근형 경향'
            else:
                face_shape = '각진형'
                confidence = 74
                reasoning = f'각진형 경향'
        else:  # 중간값
            if ratio_FW_CW > 1.02:
                face_shape = '하트형'
                confidence = 73
                reasoning = f'약한 하트형'
            elif ratio_CW_JW > 1.08:
                face_shape = '다이아몬드형'
                confidence = 71
                reasoning = f'약한 다이아몬드형'
            else:
                face_shape = '타원형'
                confidence = 75
                reasoning = f'표준 타원형'
    
    print(f"🎯 분류 결과: {face_shape} ({confidence}%) - {reasoning}")
    
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
