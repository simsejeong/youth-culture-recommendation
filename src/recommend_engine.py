# -*- coding: utf-8 -*-
"""
문화 콘텐츠 추천 엔진

입력 : 사용자 페르소나(문화 성향 · 고립·은둔 수준 · 거리/예산 응답 · 위치)
처리 : ① 거리 필터 → ② 예산 필터 → ③ 문화 성향 후보 풀 가점
       → ④ 고립·은둔 수준별 활동 강도 필터·가중치 → ⑤ 최종 점수 → ⑥ 카테고리 다양성 제한
출력 : 추천 콘텐츠 Top N + 매칭 사유
"""

import math
import numpy as np
import pandas as pd

# 설문 응답 텍스트 -> 필터 값 매핑 (Q17, Q18 보기 원문 그대로)

DISTANCE_KM_MAP = {
    "무조건 집 근처! 걸어서 10~15분 이내": 1.2,
    "대중교통으로 30분 이내": 10.0,
    "대중교통으로 1시간 내외까지 괜찮다": 20.0,
    "거리는 상관없음": None,  # None = 거리 필터를 적용하지 않음
}
 
MAX_PRICE_MAP = {
    "무료 프로그램만 선호": 0,
    "1만원 미만": 10000,
    "1~3만원 미만": 30000,
    "5만원이상도 마음에 들면 상관없음": None,  # None = 가격 상한 없음
}


# 거리 개념이 없는 카테고리 (좌표가 없어 거리 필터 · 거리 감점에서 제외)
ONLINE_CONTENT_CATEGORIES = {"온라인콘텐츠", "DIY키트"}

# 문화 성향(군집) -> 콘텐츠 후보 풀
# 풀 안의 카테고리는 +50점, 풀 밖은 +10점 (지역에 따라 풀 안 콘텐츠가 부족할 수 있어 완전 배제하지 않음)
# 영화관 · 문화예술교육프로그램은 감각형 예술 장르 중심이라 문화예술형에 포함
CLUSTER_POOL = {
    "집콕힐링형": ["나혼자문화시설", "도서관", "박물관", "미술관", "문화의집", "전시관", "온라인콘텐츠", "DIY키트"],
    "야외활동형": ["도시공원", "국공립공원", "치유의숲", "휴양림", "주말농장",
                  "농어촌체험마을", "캠핑장", "관광지", "도보여행지(골목/스토리)"],
    "문화예술형": ["박물관", "미술관", "전시관", "공연/행사", "향토문화유적",
                  "도보여행지(골목/스토리)", "영화관", "문화예술교육프로그램"]
}

POOL_BONUS = 50


# 카테고리 다양성 제한용 그룹 묶음 ---------------------------------------
# 중분류가 달라도 사용자 입장에서는 같은 종류로 보이는 경우를 하나로 묶음
# - 도시공원 / 국공립공원 → 공원류
#   (부산 시내 은둔 테스트에서 서로 다른 중분류의 공원 3곳이 Top 3를 채운 사례 발견)
# - 향토문화유적 / 도보여행지(골목/스토리) → 문화유적류 ("역사·스토리 산책"류로 비슷하게 인식)
# 다양성 체크에만 적용되며 CLUSTER_POOL · TAG_MAP 등 다른 로직에는 영향 없음
# [한계] 도보여행지에는 "전주 삼양다방"처럼 스토리가 있는 카페·빵집도 섞여 있어
#        시설명/설명 키워드로 유적류만 따로 묶으면 더 정교해질 수 있음
DIVERSITY_GROUP_MAP = {
    "도시공원": "공원류",
    "국공립공원": "공원류",
    "향토문화유적": "문화유적류",
    "도보여행지(골목/스토리)": "문화유적류",
}
 
def diversity_group(mid_category):
    return DIVERSITY_GROUP_MAP.get(mid_category, mid_category)
 
NOT_IN_POOL_BONUS = 10


# 고립은둔수준 -> 허들 강도
# max_social_score : social_intensity_score 상한 (이걸 넘는 콘텐츠는 제외)
# program_bonus    : is_program=True 콘텐츠에 더하는 가중치 (은둔은 감점, 일반은 가점)
ISOLATION_PARAMS = {
    "은둔": {"max_social_score": 1, "program_bonus": -30},
    "고립": {"max_social_score": 2, "program_bonus": 0},
    "일반": {"max_social_score": 3, "program_bonus": 30},
}

# 활동 강도 적합도 (연속 가점)
# 상한선 통과/탈락만으로는 같은 상한선 안에서 고립·은둔 수준이 순위에 영향을 주지 못함
# → 수준에 맞는 강도를 더 선호하도록 연속 가점 추가 (은둔=저강도 선호, 일반=고강도 선호)
# 고립은 은둔→고립→일반의 회복 순서를 따라 일반 방향으로 절반 가중치
# [주의] 절반(0.5) 비율은 검증된 값이 아닌 초기 기본값 → 실사용 로그로 보정 필요
SOCIAL_FIT_WEIGHT = 15
ISOLATION_FIT_RATIO = {"은둔": -1.0, "고립": 0.5, "일반": 1.0}
# -1.0 = (1-score) 방향(저강도 선호), +1.0 = score 방향(고강도 선호), 0.5는 그 절반 크기로 고강도 쪽을 약하게 선호
 
def social_fit_bonus(score, level):
    if pd.isna(score):
        return 0
        
    ratio = ISOLATION_FIT_RATIO.get(level, 0)
    
    if ratio < 0:
        return (1 - score) * SOCIAL_FIT_WEIGHT * abs(ratio)
    
    return score * SOCIAL_FIT_WEIGHT * ratio


# 무거운 역사 콘텐츠 감점 (고문·학살·항쟁 등)
# 고립 단계에 정서적으로 무거운 장소가 "가볍게 함께해볼 활동"으로 추천된 사례를 보고 추가
# 완전 제외 대신 큰 감점 → 대안이 없는 지역에서는 최후 후보로 남을 수 있음
# 키워드는 발견 사례 기반 휴리스틱이라 지속적으로 보완 필요
HEAVY_HISTORY_KEYWORDS = ["고문", "대공분실", "학살", "순국", "항쟁", "의거", "추모", "희생자"]
HEAVY_HISTORY_PENALTY = {"은둔": -100, "고립": -60, "일반": 0}
 
def heavy_history_penalty(row, level):
    text = f"{row.get('facility_name', '')} {row.get('description', '')}"
    if any(k in str(text) for k in HEAVY_HISTORY_KEYWORDS):
        return HEAVY_HISTORY_PENALTY.get(level, 0)
        
    return 0


# Haversine 거리 계산 (km)
def haversine_km(lat1, lon1, lat2, lon2):
    if any(pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
        return np.nan
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# 매칭 사유 동적 생성
# 시설 수 × 9개 조합만큼 문구를 미리 저장하지 않고,
# 추천 시점에 (문화 성향, 고립·은둔 수준, 후보 풀 포함 여부)로 생성
def get_match_reason(row, persona):
    cluster = persona["cluster"]
    level = persona["isolation_level"]
    pool = CLUSTER_POOL.get(cluster, [])
    in_pool = row.get("mid_category") in pool
 
    if not in_pool:
        return "거리·예산 조건에 맞아 추천드려요."
 
    if level == "은둔":
        reason = "혼자서도 부담 없이 둘러볼 수 있는 공간이에요."
    elif level == "고립":
        reason = "큰 부담 없이 가볍게 함께해볼 수 있는 활동이에요."
    else:
        reason = "새로운 사람들과 어울리기 좋은 활동이에요."
 
    if bool(row.get("is_program")) and level == "일반":
        reason += " 사회적 연결을 넓혀보기에도 좋아요."
 
    return reason


# 추천 함수
def recommend(master_df, persona, top_n=5, max_per_category=2, auto_relax=True):
    """
    persona = {
        "cluster": "집콕힐링형" | "야외활동형" | "문화예술형",
        "isolation_level": "은둔" | "고립" | "일반",
        "distance_answer": str,   # Q17 응답 원문
        "price_answer": str,      # Q18 응답 원문
        "user_lat": float | None, # 위치 동의 안 했으면 None
        "user_lon": float | None,
    }
    top_n            : 추천 개수
    max_per_category : 같은 카테고리 묶음이 top_n 안에서 넘지 않을 최대 개수
    auto_relax       : 결과가 0건이면 고립·은둔 허들을 1단계 완화해 재시도
                       (빈 화면보다 약간 강도 높은 대안을 보여주는 편이 낫다는 판단)
    """
    df = master_df.copy()

    
    # --- ① 거리 필터 (사용자 응답 기준) ---
    # 온라인 콘텐츠는 좌표가 없는 서비스라 거리 조건과 무관하게 항상 통과
    is_online = df["mid_category"].isin(ONLINE_CONTENT_CATEGORIES)
    radius_km = DISTANCE_KM_MAP.get(persona.get("distance_answer"))
    has_location = persona.get("user_lat") is not None and persona.get("user_lon") is not None
    if has_location:
        df["_distance_km"] = df.apply(
            lambda r: haversine_km(persona["user_lat"], persona["user_lon"], r["lat"], r["lon"]),
            axis=1,
        )
        if radius_km is not None:
            df = df[is_online | (df["_distance_km"].notna() & (df["_distance_km"] <= radius_km))]
    else:
        df["_distance_km"] = np.nan  # 위치 미동의 -> 거리 필터/정렬 보조 모두 생략
 
    # --- ② 예산 필터 (사용자 응답 기준) ---
    max_price = MAX_PRICE_MAP.get(persona.get("price_answer"))
    
    if max_price is not None:
        # 가격 정보가 없는 곳(price_known=False)은 "확인 필요"로 통과시키고 배제하지 않음
        df = df[(~df["price_known"].astype(bool)) | (df["price"].fillna(0) <= max_price)]
 
    # --- ③ 문화성향 -> 콘텐츠 후보 풀 가점 ---
    pool = CLUSTER_POOL.get(persona["cluster"], [])
    df["_pool_score"] = df["mid_category"].apply(lambda m: POOL_BONUS if m in pool else NOT_IN_POOL_BONUS)
 
    # --- ④ 고립은둔수준 -> 허들 강도 적용 ---
    level = persona["isolation_level"]
    params = ISOLATION_PARAMS.get(level, ISOLATION_PARAMS["고립"])
    filtered = df[df["social_intensity_score"].fillna(0) <= params["max_social_score"]]
 
    if filtered.empty and auto_relax:
        # 결과가 없으면 허들을 1단계 완화 (은둔 -> 고립 기준으로) 해서 재시도
        relax_order = ["은둔", "고립", "일반"]
        idx = relax_order.index(level) if level in relax_order else 1
        
        if idx < 2:
            relaxed_params = ISOLATION_PARAMS[relax_order[idx + 1]]
            filtered = df[df["social_intensity_score"].fillna(0) <= relaxed_params["max_social_score"]]
            print(f"[안내] '{level}' 기준 추천 결과가 없어 '{relax_order[idx+1]}' 기준으로 완화")
 
    df = filtered
    df["_program_score"] = df["is_program"].apply(lambda p: params["program_bonus"] if p else 0)
    df["_fit_score"] = df["social_intensity_score"].apply(lambda s: social_fit_bonus(s, level))
    df["_heavy_penalty"] = df.apply(lambda r: heavy_history_penalty(r, level), axis=1)
 
    # --- ⑤ 최종 스코어링 ---
    df["_final_score"] = df["_pool_score"] + df["_program_score"] + df["_fit_score"] + df["_heavy_penalty"]
    # 거리 감점 (1km당 -0.5점)
    # 위치 동의 사용자에게 좌표 결측 시설은 "멀다"고 가정해 감점 (결측 시설이 0km로 1등이 되는 문제 방지)
    # 온라인 콘텐츠는 거리 개념이 없으므로 감점 제외
    is_online = df["mid_category"].isin(ONLINE_CONTENT_CATEGORIES)
    distance_penalty_fill = 9999 if has_location else 0
    distance_term = df["_distance_km"].fillna(distance_penalty_fill) * 0.5
    distance_term = distance_term.where(~is_online, 0)
    df["_final_score"] -= distance_term
    df_sorted = df.sort_values("_final_score", ascending=False)
 
    
    # --- ⑥ 카테고리 다양성 제한 (그리디 선택) ---
    # 점수 순으로 선택하되 같은 카테고리 묶음이 max_per_category를 넘으면 보류
    # 후보가 부족한 지역은 보류 목록에서 점수 순으로 채워 항상 top_n개를 반환
    selected_idx, backfill_idx, category_counts = [], [], {}
    for idx, row in df_sorted.iterrows():
        if len(selected_idx) >= top_n:
            break
        cat = diversity_group(row["mid_category"])
        if category_counts.get(cat, 0) < max_per_category:
            selected_idx.append(idx)
            category_counts[cat] = category_counts.get(cat, 0) + 1
        else:
            backfill_idx.append(idx)
    if len(selected_idx) < top_n:
        selected_idx += backfill_idx[: top_n - len(selected_idx)]
 
    result = df_sorted.loc[selected_idx]
    result = result.drop(
        columns=["_distance_km", "_pool_score", "_program_score", "_fit_score", "_heavy_penalty", "_final_score"],
        errors="ignore",
    )
    result["match_reason"] = result.apply(lambda r: get_match_reason(r, persona), axis=1)
    return result
 
 
if __name__ == "__main__":
    master = pd.read_csv("recommend_master_db.csv", low_memory=False)

    # 예시: 집콕힐링형 + 은둔, 거리 "1시간 내외" / 예산 "5만원 이상도 괜찮음"
    sample_persona = {
        "cluster": "집콕힐링형",
        "isolation_level": "은둔",
        "distance_answer": "대중교통으로 1시간 내외까지 괜찮다",
        "price_answer": "5만원이상도 마음에 들면 상관없음",
        "user_lat": 37.5665,
        "user_lon": 126.9780,
    }
    recs = recommend(master, sample_persona, top_n=5)
    print(recs[["content_id", "facility_name", "mid_category", "price",
                "social_intensity_score", "match_reason"]])
