def classify_face_shape_scientific_html_logic(measurements):
    """HTML 논문 기반 얼굴형 분류 로직 - 임계값 조정으로 다양성 향상"""
    
    forehead_width = measurements["foreheadWidth"]
    cheekbone_width = measurements["cheekboneWidth"] 
    jaw_width = measurements["jawWidth"]
    face_length = measurements["faceLength"]
    
    # HTML과 동일한 핵심 비율들
    ratio_FC = face_length / cheekbone_width
    ratio_FW_CW = forehead_width / cheekbone_width
    ratio_CW_JW = cheekbone_width / jaw_width
    
    print(f"🧮 비율 계산: FC={ratio_FC:.3f}, FW_CW={ratio_FW_CW:.3f}, CW_JW={ratio_CW_JW:.3f}")
    
    # ===== 임계값 조정: 더 민감하게 분류 =====
    
    # 하트형: 이전 1.07 → 1.03으로 완화 (더 많은 하트형)
    if ratio_FW_CW > 1.03 and forehead_width > cheekbone_width and cheekbone_width > jaw_width:
        face_shape = '하트형'
        confidence = min(95, 75 + (ratio_FW_CW - 1.03) * 80)
        reasoning = f"이마폭/광대폭 비율: {ratio_FW_CW:.3f} > 1.03"
        
    # 다이아몬드형: 조건 완화 (더 많은 다이아몬드형) 
    elif (cheekbone_width > forehead_width and cheekbone_width > jaw_width and 
          ratio_CW_JW >= 1.07 and ratio_FW_CW < 0.98):  # 1.10→1.07, 0.95→0.98
        face_shape = '다이아몬드형'
        confidence = min(93, 73 + (ratio_CW_JW - 1.07) * 120)
        reasoning = f"광대폭이 최대, 광대폭/턱폭: {ratio_CW_JW:.3f}"
        
    # 긴형: 이전 1.5 → 1.32로 완화 (더 많은 긴형)
    elif ratio_FC > 1.32:
        face_shape = '긴형'
        confidence = min(91, 70 + (ratio_FC - 1.32) * 60)
        reasoning = f"얼굴길이/광대폭 비율: {ratio_FC:.3f} > 1.32"
        
    # 둥근형: 범위 확장 (더 많은 둥근형)
    elif (ratio_FC >= 1.0 and ratio_FC <= 1.15 and 
          abs(forehead_width - cheekbone_width) < 0.12 * cheekbone_width):  # 1.1→1.15, 0.1→0.12
        face_shape = '둥근형'
        confidence = min(89, 78 + (1.15 - ratio_FC) * 80)
        reasoning = f"얼굴길이/광대폭 비율: {ratio_FC:.3f} (1.0-1.15 범위)"
        
    # 각진형: 조건 완화 (더 많은 각진형)
    elif (ratio_FC <= 1.20 and abs(forehead_width - cheekbone_width) < 0.18 * cheekbone_width and
          abs(cheekbone_width - jaw_width) < 0.18 * cheekbone_width):  # 1.15→1.20, 0.15→0.18
        face_shape = '각진형'
        confidence = min(87, 72 + (1.20 - ratio_FC) * 80)
        reasoning = f"이마≈광대≈턱, 비율: {ratio_FC:.3f} ≤ 1.20"
        
    # 타원형: 범위 축소 (타원형 줄이기)
    elif ratio_FC >= 1.22 and ratio_FC <= 1.38:  # 1.3→1.22, 1.5→1.38
        face_shape = '타원형'
        confidence = min(92, 82 + (1.30 - abs(ratio_FC - 1.30)) * 80)
        reasoning = f"황금 비율: {ratio_FC:.3f} (1.22-1.38 범위)"
        
    else:
        # 경계 케이스 - 다양한 분류로 재배치
        if ratio_FC > 1.20:  # 긴 편
            if ratio_FW_CW > 1.02:  # 이마가 넓은 편
                face_shape = '하트형'  # 타원형 → 하트형
                confidence = 78
                reasoning = f'긴 하트형 경향 (FC:{ratio_FC:.3f}, FW_CW:{ratio_FW_CW:.3f})'
            else:
                face_shape = '긴형'  # 타원형 → 긴형
                confidence = 76
                reasoning = f'긴형 경향 (FC:{ratio_FC:.3f})'
        elif ratio_FC < 1.20:  # 짧은 편
            if abs(forehead_width - cheekbone_width) < 0.15 * cheekbone_width:
                face_shape = '둥근형'  # 타원형 → 둥근형
                confidence = 74
                reasoning = f'둥근형 경향 (FC:{ratio_FC:.3f})'
            else:
                face_shape = '각진형'  # 타원형 → 각진형
                confidence = 72
                reasoning = f'각진형 경향 (FC:{ratio_FC:.3f})'
        else:  # 중간값
            if ratio_FW_CW > 1.01:  # 1.02→1.01 더 민감하게
                face_shape = '하트형'
                confidence = 71
                reasoning = f'약한 하트형 (FW_CW:{ratio_FW_CW:.3f})'
            elif ratio_CW_JW > 1.06:  # 1.08→1.06 더 민감하게
                face_shape = '다이아몬드형'
                confidence = 69
                reasoning = f'약한 다이아몬드형 (CW_JW:{ratio_CW_JW:.3f})'
            else:
                face_shape = '각진형'  # 타원형 → 각진형
                confidence = 70
                reasoning = f'균형잡힌 각진형'
    
    print(f"🎯 조정된 분류: {face_shape} ({confidence}%) - {reasoning}")
    
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
