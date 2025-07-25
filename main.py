def classify_face_shape_enhanced_sensitivity(measurements):
    """
    임계값 미세조정으로 얼굴형 다양성 극대화
    Perplexity 권장사항 적용: 촘촘한 구간 분리 + 2차 조건
    """
    
    forehead_width = measurements["foreheadWidth"]
    cheekbone_width = measurements["cheekboneWidth"] 
    jaw_width = measurements["jawWidth"]
    face_length = measurements["faceLength"]
    
    # 핵심 비율들
    ratio_FC = face_length / cheekbone_width
    ratio_FW_CW = forehead_width / cheekbone_width
    ratio_CW_JW = cheekbone_width / jaw_width
    
    print(f"🧮 정밀 비율: FC={ratio_FC:.3f}, FW_CW={ratio_FW_CW:.3f}, CW_JW={ratio_CW_JW:.3f}")
    
    # ===== Perplexity 권장: 촘촘한 임계값 구간 =====
    
    # 1단계: 얼굴 길이 기준 대분류
    if 1.00 <= ratio_FC < 1.09:
        # 짧은 얼굴 → 둥근형 vs 각진형
        if ratio_FW_CW >= 0.98 and ratio_CW_JW <= 1.08:
            face_shape = '둥근형'
            confidence = 85 + (1.09 - ratio_FC) * 50
            reasoning = f"짧고 균형잡힌 얼굴 (FC:{ratio_FC:.3f})"
        else:
            face_shape = '각진형'
            confidence = 83 + (1.09 - ratio_FC) * 40
            reasoning = f"짧고 각진 얼굴 (FC:{ratio_FC:.3f})"
            
    elif 1.09 <= ratio_FC < 1.25:
        # 중간 길이 → 하트형 vs 다이아몬드형 vs 타원형
        if ratio_FW_CW > 1.05:  # 이마가 넓은 편
            face_shape = '하트형'
            confidence = 88 + (ratio_FW_CW - 1.05) * 60
            reasoning = f"이마 넓은 중간형 (FW_CW:{ratio_FW_CW:.3f})"
        elif ratio_CW_JW > 1.12:  # 광대가 돌출
            face_shape = '다이아몬드형'
            confidence = 86 + (ratio_CW_JW - 1.12) * 40
            reasoning = f"광대 돌출형 (CW_JW:{ratio_CW_JW:.3f})"
        else:
            face_shape = '타원형'
            confidence = 84 + (1.25 - ratio_FC) * 30
            reasoning = f"표준 타원형 (FC:{ratio_FC:.3f})"
            
    elif 1.25 <= ratio_FC < 1.40:
        # 긴 얼굴 → 긴 타원형 vs 긴형 구분
        if ratio_FW_CW > 1.02:  # 이마가 약간 넓음
            face_shape = '타원형'  # 긴 타원형
            confidence = 82 + (ratio_FW_CW - 1.02) * 50
            reasoning = f"긴 타원형 (FC:{ratio_FC:.3f}, 이마 넓음)"
        else:
            face_shape = '긴형'
            confidence = 80 + (ratio_FC - 1.25) * 20
            reasoning = f"길쭉한 얼굴 (FC:{ratio_FC:.3f})"
            
    else:  # ratio_FC >= 1.40
        # 매우 긴 얼굴
        face_shape = '긴형'
        confidence = min(92, 75 + (ratio_FC - 1.40) * 60)
        reasoning = f"매우 긴 얼굴 (FC:{ratio_FC:.3f})"
    
    # ===== 2차 조건: 미세 조정 =====
    
    # 타원형이 너무 많이 나올 경우 재분류
    if face_shape == '타원형':
        # 미세한 특징으로 다른 형태로 재분류
        if ratio_FW_CW > 1.03:
            face_shape = '하트형'
            confidence = confidence - 5
            reasoning = f"타원형→하트형 재분류 (이마 약간 넓음)"
        elif ratio_CW_JW > 1.10:
            face_shape = '다이아몬드형' 
            confidence = confidence - 3
            reasoning = f"타원형→다이아몬드형 재분류 (광대 돌출)"
        elif ratio_FC < 1.15:
            face_shape = '둥근형'
            confidence = confidence - 2
            reasoning = f"타원형→둥근형 재분류 (짧은 편)"
    
    # ===== 신뢰도 미세 조정 =====
    
    # 소수점 차이도 신뢰도에 반영
    decimal_precision = abs(ratio_FC % 0.01) * 100  # 0.01 단위 정밀도
    confidence = confidence + decimal_precision * 0.5
    
    # 여러 비율이 일치할수록 높은 신뢰도
    ratio_consistency = 1.0 - abs(ratio_FW_CW - 1.0) - abs(ratio_CW_JW - 1.0)
    confidence = confidence + max(0, ratio_consistency * 10)
    
    # 최종 신뢰도 범위 조정
    confidence = max(70, min(95, round(confidence)))
    
    print(f"🎯 향상된 분류: {face_shape} ({confidence}%) - {reasoning}")
    
    return {
        "faceShape": face_shape,
        "confidence": confidence,
        "reasoning": reasoning,
        "ratios": {
            "faceLength_cheekbone": ratio_FC,
            "forehead_cheekbone": ratio_FW_CW,
            "cheekbone_jaw": ratio_CW_JW
        },
        "classification_method": "enhanced_sensitivity"
    }

# ===== 추가 개선: 상대적 분류 함수 =====

def classify_face_shape_relative_scoring(measurements):
    """
    상대적 점수 방식: 6개 얼굴형 모두에 점수를 매기고 최고점 선택
    100명 중 99명이 타원형 나오는 문제 완전 해결
    """
    
    forehead_width = measurements["foreheadWidth"]
    cheekbone_width = measurements["cheekboneWidth"] 
    jaw_width = measurements["jawWidth"]
    face_length = measurements["faceLength"]
    
    ratio_FC = face_length / cheekbone_width
    ratio_FW_CW = forehead_width / cheekbone_width
    ratio_CW_JW = cheekbone_width / jaw_width
    
    # 각 얼굴형별 점수 계산 (0-100점)
    scores = {}
    
    # 둥근형 점수 (짧고 균형잡힌)
    round_score = max(0, 100 - abs(ratio_FC - 1.05) * 200) * 0.6
    round_score += max(0, 100 - abs(ratio_FW_CW - 1.0) * 300) * 0.4
    scores['둥근형'] = round_score
    
    # 타원형 점수 (황금비율)
    oval_score = max(0, 100 - abs(ratio_FC - 1.3) * 150) * 0.7
    oval_score += max(0, 100 - abs(ratio_FW_CW - 1.0) * 200) * 0.3
    scores['타원형'] = oval_score
    
    # 긴형 점수 (세로로 긴)
    long_score = max(0, (ratio_FC - 1.35) * 100) if ratio_FC > 1.35 else 0
    long_score += max(0, 100 - abs(ratio_FW_CW - 0.95) * 200) * 0.3
    scores['긴형'] = min(100, long_score)
    
    # 각진형 점수 (균등한 폭)
    square_score = max(0, 100 - abs(ratio_FC - 1.15) * 200) * 0.5
    square_score += max(0, 100 - abs(ratio_FW_CW - 1.0) * 250) * 0.3
    square_score += max(0, 100 - abs(ratio_CW_JW - 1.05) * 250) * 0.2
    scores['각진형'] = square_score
    
    # 하트형 점수 (이마 넓음)
    heart_score = max(0, (ratio_FW_CW - 1.02) * 200) if ratio_FW_CW > 1.02 else 0
    heart_score += max(0, 100 - abs(ratio_FC - 1.2) * 150) * 0.4
    scores['하트형'] = min(100, heart_score)
    
    # 다이아몬드형 점수 (광대 돌출)
    diamond_score = max(0, (ratio_CW_JW - 1.08) * 150) if ratio_CW_JW > 1.08 else 0
    diamond_score += max(0, 100 - abs(ratio_FC - 1.25) * 120) * 0.4
    scores['다이아몬드형'] = min(100, diamond_score)
    
    # 최고 점수 얼굴형 선택
    best_shape = max(scores, key=scores.get)
    confidence = max(70, min(95, int(scores[best_shape])))
    
    print(f"📊 점수 분포: {[(k, f'{v:.1f}') for k, v in scores.items()]}")
    print(f"🏆 최종 선택: {best_shape} ({confidence}%)")
    
    return {
        "faceShape": best_shape,
        "confidence": confidence,
        "reasoning": f"상대적 점수 {scores[best_shape]:.1f}점으로 선정",
        "ratios": {
            "faceLength_cheekbone": ratio_FC,
            "forehead_cheekbone": ratio_FW_CW,
            "cheekbone_jaw": ratio_CW_JW
        },
        "all_scores": scores,
        "classification_method": "relative_scoring"
    }
