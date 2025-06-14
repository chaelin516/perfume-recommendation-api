import os
import math
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Literal
import pandas as pd
import json

# ─── 로거 설정 ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("course_router")

# ─── 1. perfume_final_dataset.csv 로드 ───────────────────────────────────────────────
PERFUME_CSV_PATH = os.path.join(os.path.dirname(__file__), "../data/perfume_final_dataset.csv")
try:
    perfume_df = pd.read_csv(PERFUME_CSV_PATH)
    perfume_df.fillna("", inplace=True)
    logger.info(f"Perfume dataset loaded: {perfume_df.shape[0]} rows, columns: {list(perfume_df.columns)}")

    # ✅ 실제 컬럼 확인 및 샘플 로그
    if not perfume_df.empty:
        sample_row = perfume_df.iloc[0]
        logger.info("=== DATASET SAMPLE INFO ===")
        logger.info(f"Sample perfume: {sample_row['name']} by {sample_row['brand']}")
        logger.info(f"Sample gender: '{sample_row['gender']}'")
        logger.info(f"Sample season_tags: '{sample_row['season_tags']}'")
        logger.info(f"Sample time_tags: '{sample_row['time_tags']}'")

        # ✅ emotion_cluster 컬럼 사용 (emotion_tags 대신)
        if 'emotion_cluster' in perfume_df.columns:
            logger.info(f"Sample emotion_cluster: '{sample_row['emotion_cluster']}'")

        # ✅ desired_impression 컬럼 확인
        if 'desired_impression' in perfume_df.columns:
            logger.info(f"Sample desired_impression: '{sample_row['desired_impression']}'")

except Exception as e:
    logger.error(f"perfume_final_dataset.csv 로드 중 오류: {e}")
    raise RuntimeError(f"perfume_final_dataset.csv 로드 중 오류: {e}")

# ─── 2. store_data.json 로드 ─────────────────────────────────────────────────────────
STORE_JSON_PATH = os.path.join(os.path.dirname(__file__), "../data/store_data.json")
try:
    with open(STORE_JSON_PATH, "r", encoding="utf-8") as f:
        store_data = json.load(f)
    logger.info(f"Store data loaded: {len(store_data)} entries")
except Exception as e:
    logger.error(f"store_data.json 로드 중 오류: {e}")
    raise RuntimeError(f"store_data.json 로드 중 오류: {e}")


# ─── 3. 성별 매핑 함수 ───────────────────────────────────────────────────────────────
def map_gender_for_perfume(request_gender: str) -> str:
    """
    요청에서 받은 'male'/'female' 값을 perfume_df의 'men'/'women' 컬럼 값으로 매핑합니다.
    'unisex'는 그대로 'unisex'로 유지합니다.
    """
    if request_gender == "female":
        return "women"
    if request_gender == "male":
        return "men"
    return "unisex"


# ─── 4. 거리 계산 (Haversine) 함수 ──────────────────────────────────────────────────
def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    두 위도/경도 지점 간의 거리를 킬로미터 단위로 계산합니다.
    """
    R = 6371.0  # 지구 반경 (km)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# ─── 5. 감정 매핑 함수 ──────────────────────────────────────────────────────────────
def get_emotion_from_impression(desired_impression: str, request_emotion: str) -> bool:
    """
    desired_impression 문자열에서 요청된 감정이 포함되어 있는지 확인합니다.
    """
    if not desired_impression or not request_emotion:
        return False

    # desired_impression은 "confident, fresh" 같은 형태로 되어 있음
    impressions = [imp.strip().lower() for imp in desired_impression.split(",")]
    return request_emotion.lower() in impressions


# ─── 6. 요청(Request) 스키마 정의 ────────────────────────────────────────────────────
class CourseRecommendRequest(BaseModel):
    gender: Literal["male", "female", "unisex"]
    emotion: Literal["confident", "elegant", "pure", "friendly", "mysterious", "fresh"]
    season: Literal["spring", "summer", "fall", "winter"]
    time: Literal["day", "night"]
    activity: Literal["casual", "work", "date"] = Field(
        ..., description="현재는 필터링에 사용되지 않음"
    )
    latitude: float = Field(..., description="사용자 현재 위도")
    longitude: float = Field(..., description="사용자 현재 경도")


# ─── 7. 코스 추천 결과 아이템 스키마 ───────────────────────────────────────────────
class CourseItem(BaseModel):
    store_name: str
    store_address: str
    store_latitude: float
    store_longitude: float
    recommended_perfumes: List[str]


class CourseRecommendResponse(BaseModel):
    message: str
    data: List[CourseItem]


router = APIRouter(prefix="/courses", tags=["Course"])


@router.post(
    "/recommend",
    response_model=CourseRecommendResponse,
    summary="향수 코스 추천",
    description=(
            "gender, emotion, season, time, activity, latitude, longitude 정보를 받아\n"
            "1) perfume_final_dataset.csv에서 조건(gender, emotion, season, time)에 맞는 향수를 필터링,\n"
            "2) 상위 5개의 향수를 선택하고,\n"
            "3) 사용자의 위치로부터 반경 1.5km 이내 매장을 찾아 해당 매장별로 추천 향수 목록을 묶어서 반환합니다.\n\n"
            "※ activity 필드는 현재 필터링에 사용되지 않으며, emotion은 desired_impression 컬럼을 통해 매칭됩니다."
    )
)
def recommend_course(request: CourseRecommendRequest):
    logger.info(f"[COURSE] 요청 파라미터: "
                f"gender={request.gender}, emotion={request.emotion}, season={request.season}, "
                f"time={request.time}, activity={request.activity}, "
                f"latitude={request.latitude}, longitude={request.longitude}")

    # 1) 성별 매핑: 'female' → 'women', 'male' → 'men', 'unisex' → 'unisex'
    mapped_gender = map_gender_for_perfume(request.gender)
    logger.info(f"[COURSE] 매핑된 gender: {mapped_gender}")

    # 2) perfume 후보 필터링
    #    실제 perfume_final_dataset.csv 컬럼: ['name','brand','image_url','gender','notes',
    #                                        'season_tags','time_tags','brand_tag','activity',
    #                                        'weather','desired_impression','emotion_cluster']
    try:
        # ✅ 기본 필터링: gender, season, time
        candidates = perfume_df[
            (perfume_df["gender"] == mapped_gender)
            & (perfume_df["season_tags"].str.contains(request.season, na=False))
            & (perfume_df["time_tags"].str.contains(request.time, na=False))
            ]

        logger.info(f"[COURSE] 기본 필터링 후 향수 개수: {len(candidates)}")

        # ✅ emotion 필터링 (desired_impression 컬럼 사용)
        if 'desired_impression' in perfume_df.columns and not candidates.empty:
            emotion_filtered = []
            for idx, row in candidates.iterrows():
                if get_emotion_from_impression(row['desired_impression'], request.emotion):
                    emotion_filtered.append(idx)

            if emotion_filtered:
                candidates = candidates.loc[emotion_filtered]
                logger.info(f"[COURSE] emotion 필터링 후 향수 개수: {len(candidates)}")
            else:
                logger.warning(f"[COURSE] emotion '{request.emotion}' 필터링 결과 향수 없음")

    except Exception as e:
        logger.error(f"[COURSE] 향수 후보 필터링 오류: {e}")
        raise HTTPException(status_code=500, detail=f"향수 후보 필터링 중 오류가 발생했습니다: {e}")

    logger.info(f"[COURSE] 최종 필터링된 향수 후보 개수: {len(candidates)}")
    if candidates.empty:
        logger.info("[COURSE] 후보 향수가 없어 빈 결과 반환")
        return CourseRecommendResponse(message="조건에 맞는 향수가 없습니다.", data=[])

    # 3) 상위 5개 향수(예시)만 추출 (필요 시 모델 점수 기반으로 수정 가능)
    top_perfumes = candidates["name"].tolist()[:5]
    logger.info(f"[COURSE] 상위 추천 향수(최대 5개): {top_perfumes}")

    # 4) 위치 기반 가까운 매장 찾기 (반경 1.5km)
    user_lat = request.latitude
    user_lng = request.longitude
    radius_km = 1.5
    nearby_stores = []

    for store in store_data:
        store_lat = store.get("lat")
        store_lng = store.get("lng")

        if store_lat is None or store_lng is None:
            logger.warning(f"[COURSE] 매장 '{store.get('name')}' 위치 정보 없음")
            continue

        distance = haversine(user_lat, user_lng, store_lat, store_lng)
        logger.info(f"[COURSE] 매장명={store.get('name')} 거리={distance:.2f}km")

        if distance <= radius_km:
            nearby_stores.append({
                "store_name": store.get("name"),
                "store_address": store.get("address"),
                "store_latitude": store_lat,
                "store_longitude": store_lng,
                "recommended_perfumes": top_perfumes
            })

    logger.info(f"[COURSE] 반경 {radius_km}km 내 매장 개수: {len(nearby_stores)}")
    if not nearby_stores:
        logger.info("[COURSE] 인근 매장이 없어 빈 결과 반환")
        return CourseRecommendResponse(message="반경 1.5km 내에 매장이 없습니다.", data=[])

    # 5) CourseItem 객체로 변환
    response_items: List[CourseItem] = []
    for store in nearby_stores:
        response_items.append(
            CourseItem(
                store_name=store["store_name"],
                store_address=store["store_address"],
                store_latitude=store["store_latitude"],
                store_longitude=store["store_longitude"],
                recommended_perfumes=store["recommended_perfumes"]
            )
        )

    logger.info(f"[COURSE] 최종 반환될 코스 개수: {len(response_items)}")
    return CourseRecommendResponse(message="향수 코스 추천 성공", data=response_items)