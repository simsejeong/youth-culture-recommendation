# -*- coding: utf-8 -*-
"""
문화 성향 분류기

1) classify_by_rule        : 최종 분류 로직 — Q14(활동 장소) · Q16(힐링 장소) 2문항 규칙
2) classify_culture_cluster : 비교용 — K-means(k=3) 모델로 분류 (팀 배포 버전)

[K-means 분류기]
새 응답자의 Q9·Q14·Q15·Q16·Q19(단일 선택) + Q20(복수 선택) 답변을
설문 분석 노트북에서 학습·저장한 OneHotEncoder + KMeans(k=3)로 분류합니다.
학습 때와 같은 인코딩·순서로 변환하므로 사람이 만든 규칙 없이 동일한 기준으로 분류됩니다.

[필요 파일] notebooks/01_survey_clustering.ipynb 실행 시 models/ 폴더에 생성
    culture_ohe.joblib, culture_kmeans.joblib, q20_map.joblib, cluster_label_map.joblib
[주의] cluster_label_map의 번호→라벨 매핑은 군집 크기(79/61/41) 순서를 기준으로 함
       재학습 시 순서가 같은지 확인 필요
"""
 
import os
import numpy as np
import joblib

ARTIFACT_DIR = os.environ.get(
    "CULTURE_MODEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"),
)  # joblib 파일 위치 (기본: 저장소의 models/ 폴더)

# =====================================================================
# 1) 2문항 규칙 기반 분류 (최종)
#    K-means 군집이 Q14·Q16에 크게 의존함을 확인하고, 3개 유형을 유지하면서
#    해석 가능한 규칙으로 단순화
#      Q14 집안 선호                    → 집콕힐링형
#      Q14 야외 선호 + Q16 숲·공원       → 야외활동형
#      Q14 야외 선호 + Q16 갤러리·북카페 → 문화예술형
# =====================================================================
Q14_INDOOR = "온라인, 집안"
Q16_NATURE = "피톤치드 가득한 숲속이나 탁 트인 조용한 공원"


def classify_by_rule(q14_place_pref, q16_healing_spot):
    """Q14 · Q16 응답 원문을 받아 집콕힐링형 / 야외활동형 / 문화예술형 중 하나를 반환"""
    if q14_place_pref == Q14_INDOOR:
        return "집콕힐링형"
    if q16_healing_spot == Q16_NATURE:
        return "야외활동형"
    return "문화예술형"


# =====================================================================
# 2) K-means(k=3) 모델 기반 분류 (비교용)
# =====================================================================

 
# 노트북의 culture_cols와 정확히 같은 순서/이름이어야 합니다 (OneHotEncoder.fit 때 순서).
CULTURE_COLS = ["q9_time", "q14_place_pref", "q15_interaction", "q16_healing_spot", "q19_stimulus_type"]
 
_ohe = None
_km = None
_q20_map = None
_label_map = None
 
 
def _load_artifacts():
    global _ohe, _km, _q20_map, _label_map
    if _ohe is None:
        _ohe = joblib.load(os.path.join(ARTIFACT_DIR, "culture_ohe.joblib"))
        _km = joblib.load(os.path.join(ARTIFACT_DIR, "culture_kmeans.joblib"))
        _q20_map = joblib.load(os.path.join(ARTIFACT_DIR, "q20_map.joblib"))
        _label_map = joblib.load(os.path.join(ARTIFACT_DIR, "cluster_label_map.joblib"))
    return _ohe, _km, _q20_map, _label_map
 
 
def classify_culture_cluster(answers):
    """
    answers = {
        "q9_time": "조명과 야경, 차분한 분위기가 더해주는 밤",
        "q14_place_pref": "온라인, 집안",
        "q15_interaction": "누구의 방해도 없이 온전히 나 혼자 집중하고 싶다",
        "q16_healing_spot": "에어컨이 시원하고 세련된 인테리어의 조용한 갤러리나 북카페",
        "q19_stimulus_type": "생각과 해석이 필요한 활동 (독서 토론·영화 비평·글쓰기·연극 관람)",
        "q20_activities": ["혼자 실내", "공연·전시"],   # Q20 체크박스에서 선택한 값들 (중복 가능)
    }
    -> "집콕힐링형" | "야외활동형" | "문화예술형"
    """
    ohe, km, q20_map, label_map = _load_artifacts()
 
    # 1) 단일선택 5문항 -> OneHotEncoder.transform (학습 때와 같은 컬럼 순서로 한 줄 입력)
    single_choice_row = [[answers[col] for col in CULTURE_COLS]]
    x_ohe = ohe.transform(single_choice_row)  # handle_unknown='ignore'라 모르는 보기는 전부 0으로 처리됨
 
    # 2) Q20(복수 응답) -> q20_map의 키 순서대로 9개 0/1 벡터
    #    앱에서는 짧은 라벨(공방·DIY 등)로 받으므로 짧은 라벨 <-> 원래 키를 매칭
    short_labels = {
        "q20_craft": "공방·DIY", "q20_concert": "공연·전시", "q20_outdoor": "야외·자연",
        "q20_cafe": "혼자 실내", "q20_heritage": "역사·전통", "q20_sport": "스포츠",
        "q20_fun": "오락·게임", "q20_modern": "현대미술", "q20_classic": "클래식·예술",
    }
    selected = set(answers.get("q20_activities", []))
    x_q20 = np.array([[1 if short_labels[k] in selected else 0 for k in q20_map.keys()]])
 
    # 3) 합쳐서 예측 (학습 때 np.hstack([X_ohe, X_q20])과 정확히 같은 순서)
    x_new = np.hstack([x_ohe, x_q20])
    cluster_num = int(km.predict(x_new)[0])
 
    return label_map[cluster_num]
 
 
if __name__ == "__main__":
    sample_answers = {
        "q9_time": "조명과 야경, 차분한 분위기가 더해주는 밤",
        "q14_place_pref": "온라인, 집안",
        "q15_interaction": "누구의 방해도 없이 온전히 나 혼자 집중하고 싶다",
        "q16_healing_spot": "에어컨이 시원하고 세련된 인테리어의 조용한 갤러리나 북카페",
        "q19_stimulus_type": "생각과 해석이 필요한 활동 (독서 토론·영화 비평·글쓰기·연극 관람)",
        "q20_activities": ["혼자 실내", "공연·전시"],
    }
    print("규칙 기반 :", classify_by_rule(sample_answers["q14_place_pref"], sample_answers["q16_healing_spot"]))
    print("K-means   :", classify_culture_cluster(sample_answers))