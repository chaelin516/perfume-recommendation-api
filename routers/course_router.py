import os
import math
import logging
import pickle
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Literal
import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import load_model

# ─── 로거 설정 ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("course_router")

# ─── 1. 파일 경로 설정 ───────────────────────────────────────────────
PERFUME_CSV_PATH = os.path.join(os.path.dirname(__file__), "../data/perfume_final_dataset.csv")
STORE_JSON_PATH = os.path.join(os.path.dirname(__file__), "../data/store_data.json")


# ─── 2. 데이터 로딩 함수들 (lazy loading) ─────────────────────────────────────────
def load_perfume_data():
    """향수 데이터 로드 (필요시에만)"""
    try:
        perfume_df = pd.read_csv(PERFUME_CSV_PATH)
        perfume_df.fillna("", inplace=True)
        logger.info(f"Perfume dataset loaded: {perfume_df.shape[0]} rows, columns: {list(perfume_df.columns)}")
        return perfume_df
    except Exception as e:
        logger.error(f"perfume_final_dataset.csv 로드 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"향수 데이터 로드 실패: {e}")


def load_store_data():
    """매장 데이터 로드 (필요시에만)"""
    try:
        with open(STORE_JSON_PATH, "r", encoding="utf-8") as f:
            store_data = json.load(f)
        logger.info(f"Store data loaded: {len(store_data)} entries")
        return store_data
    except Exception as e:
        logger.error(f"store_data.json 로드 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"매장 데이터 로드 실패: {e}")


# ─── 3. 모델 및 인코더 로드 (recommend_router.py와 동일한 로직) ─────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/final_model_perfume.keras")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "../models/encoder.pkl")

# 전역 변수(lazy loading)
_model = None


def get_model():
    global _model
    if _model is None:
        try:
            _model = load_model(MODEL_PATH)
            logger.info("Keras 모델 로드 성공")
        except Exception as e:
            logger.error(f"Keras 모델 로드 중 오류: {e}")
            raise HTTPException(status_code=500, detail=f"AI 모델 로드 실패: {e}")
    return _model


# encoder.pkl 로드 함수
def get_saved_encoder():
    try:
        with open(ENCODER_PATH, "rb") as f:
            encoder = pickle.load(f)
        logger.info("encoder.pkl 로드 성공")
        return encoder
    except Exception as e:
        logger.error(f"encoder.pkl 로드 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"인코더 로드 실패: {e}")


# fallback OneHotEncoder: 카테고리 직접 선언, handle_unknown="ignore" 설정
CATEGORIES = [
    ["women", "men", "unisex"],  # gender
    ["spring", "summer", "fall", "winter"],  # season
    ["day", "night"],  # time
    ["confident", "elegant", "pure", "friendly", "mysterious", "fresh"],  # impression
    ["casual", "work", "date"],  # activity
    ["hot", "cold", "rainy", "any"]  # weather
]
_fallback_encoder = OneHotEncoder(categories=CATEGORIES, handle_unknown="ignore", sparse=False)
# 더미 데이터를 이용해 한 번 fit() 호출
_fallback_encoder.fit([
    ["women", "spring", "day", "confident", "casual", "hot"],
    ["men", "summer", "night", "elegant", "work", "cold"]
])


# ─── 4. 성별 매핑 함수 (수정됨) ──────────────────────────────────────────────────────
def map_gender_for_model(request_gender: str) -> str:
    """
    요청에서 받은 'male'/'female' 값을 모델이 기대하는 'men'/'women' 값으로 매핑합니다.
    """
    if request_gender == "female":
        return "women"
    elif request_gender == "male":
        return "men"
    return "unisex"


# ─── 5. 거리 계산 (Haversine) 함수 ──────────────────────────────────────────────────
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


# ─── 6. 요청(Request) 스키마 정의 (수정됨 - 모델 입력에 맞춤) ────────────────────────────
class CourseRecommendRequest(BaseModel):
    gender: Literal["male", "female", "unisex"]
    season: Literal["spring", "summer", "fall", "winter"]
    time: Literal["day", "night"]
    impression: Literal["confident", "elegant", "pure", "friendly", "mysterious", "fresh"]
    activity: Literal["casual", "work", "date"]
    weather: Literal["hot", "cold", "rainy", "any"]
    latitude: float = Field(..., description="사용자 현재 위도")
    longitude: float = Field(..., description="사용자 현재 경도")


# ─── 7. 코스 추천 결과 아이템 스키마 (개선됨) ──────────────────────────────────────────
class RecommendedPerfume(BaseModel):
    name: str
    brand: str
    image_url: str
    score: float
    reason: str


class CourseItem(BaseModel):
    store_name: str
    store_address: str
    store_latitude: float
    store_longitude: float
    distance_km: float
    recommended_perfumes: List[RecommendedPerfume]


class CourseRecommendResponse(BaseModel):
    message: str
    total_stores: int
    recommended_perfumes_count: int
    data: List[CourseItem]


router = APIRouter(prefix="/courses", tags=["Course"])


@router.post(
    "/recommend",
    response_model=CourseRecommendResponse,
    summary="AI 모델 기반 향수 코스 추천",
    description=(
            "사용자의 취향(gender, season, time, impression, activity, weather)과 위치 정보를 받아\n"
            "1) 학습된 AI 모델(final_model_perfume.keras)로 향수 추천 점수를 계산하고,\n"
            "2) 상위 점수의 향수들을 선택한 후,\n"
            "3) 사용자 위치 기준 반경 5km 이내 매장들과 매칭하여 시향 코스를 추천합니다.\n\n"
            "각 매장별로 추천 향수와 점수, 추천 이유를 제공합니다."
    )
)
def recommend_course(request: CourseRecommendRequest):
    logger.info(f"[COURSE] 요청 파라미터: "
                f"gender={request.gender}, season={request.season}, time={request.time}, "
                f"impression={request.impression}, activity={request.activity}, weather={request.weather}, "
                f"latitude={request.latitude}, longitude={request.longitude}")

    try:
        # ─── 0. 데이터 로드 ──────────────────────────────────────────────────────
        perfume_df = load_perfume_data()
        store_data = load_store_data()

        # ─── 1. 모델 입력 준비 ──────────────────────────────────────────────────────
        # 성별 매핑: 'female' → 'women', 'male' → 'men'
        mapped_gender = map_gender_for_model(request.gender)
        logger.info(f"[COURSE] 매핑된 gender: {mapped_gender}")

        # 모델 입력 특성 구성 (recommend_router.py와 동일한 순서)
        raw_features = [
            mapped_gender,  # gender
            request.season,  # season
            request.time,  # time
            request.impression,  # impression (desired_impression)
            request.activity,  # activity
            request.weather  # weather
        ]

        logger.info(f"[COURSE] 모델 입력 특성: {raw_features}")

        # ─── 2. 특성 인코딩 (encoder.pkl 우선, 실패시 fallback) ──────────────────────
        use_fallback = False
        try:
            encoder = get_saved_encoder()
            x_input = encoder.transform([raw_features])
            logger.info("[COURSE] encoder.pkl 사용하여 전처리 성공")
        except Exception as e:
            logger.warning(f"[COURSE] encoder.pkl 전처리 실패: {e} → fallback 사용")
            use_fallback = True

        if use_fallback:
            try:
                x_input = _fallback_encoder.transform([raw_features])
                logger.info("[COURSE] fallback OneHotEncoder 사용하여 전처리 성공")
            except Exception as e:
                logger.error(f"[COURSE] fallback 전처리 중 오류: {e}")
                raise HTTPException(status_code=400, detail=f"입력값 전처리 중 오류: {e}")

        # ─── 3. AI 모델 예측 ───────────────────────────────────────────────────────
        try:
            model = get_model()
            predictions = model.predict(x_input)  # shape: (1, num_perfumes)
            scores = predictions.flatten()  # shape: (num_perfumes,)
            logger.info(f"[COURSE] 모델 예측 성공, 점수 범위: {scores.min():.4f} ~ {scores.max():.4f}")
        except Exception as e:
            logger.error(f"[COURSE] 모델 예측 중 오류: {e}")
            raise HTTPException(status_code=500, detail=f"모델 예측 중 오류: {e}")

        # ─── 4. 모델 출력과 데이터 크기 검증 ──────────────────────────────────────────
        if len(scores) != len(perfume_df):
            logger.error(f"[COURSE] 크기 불일치: 모델 출력 {len(scores)} vs 데이터 {len(perfume_df)}")
            raise HTTPException(
                status_code=500,
                detail="모델 출력 크기가 향수 데이터와 일치하지 않습니다."
            )

        # ─── 5. 향수 데이터에 점수 추가 및 정렬 ─────────────────────────────────────────
        perfume_with_scores = perfume_df.copy()
        perfume_with_scores["ai_score"] = scores

        # 점수 순으로 내림차순 정렬하여 상위 향수 선택
        top_perfumes = perfume_with_scores.sort_values(by="ai_score", ascending=False).head(10)
        logger.info(
            f"[COURSE] 상위 10개 향수 선정 완료 (점수: {top_perfumes['ai_score'].iloc[0]:.4f} ~ {top_perfumes['ai_score'].iloc[-1]:.4f})")

        # ─── 6. 위치 기반 매장 필터링 ──────────────────────────────────────────────────
        user_lat = request.latitude
        user_lng = request.longitude
        radius_km = 5.0  # 반경 5km로 확장

        nearby_stores = []
        for store in store_data:
            store_lat = store.get("lat")
            store_lng = store.get("lng")

            if store_lat is None or store_lng is None:
                logger.warning(f"[COURSE] 매장 {store.get('name')}의 좌표 정보 누락")
                continue

            distance = haversine(user_lat, user_lng, store_lat, store_lng)

            if distance <= radius_km:
                # 각 매장에서 판매하는 향수와 추천 향수 매칭
                store_perfumes = store.get("perfumes", [])
                matched_perfumes = []

                for _, perfume_row in top_perfumes.iterrows():
                    perfume_name = perfume_row["name"]
                    if perfume_name in store_perfumes:
                        matched_perfumes.append(RecommendedPerfume(
                            name=perfume_name,
                            brand=perfume_row["brand"],
                            image_url=perfume_row["image_url"],
                            score=round(float(perfume_row["ai_score"]), 4),
                            reason=f"AI 모델 추천 점수 {perfume_row['ai_score']:.4f} - {request.impression} 인상에 최적"
                        ))

                if matched_perfumes:  # 매칭되는 향수가 있는 매장만 추가
                    nearby_stores.append({
                        "store_name": store.get("name", "Unknown Store"),
                        "store_address": store.get("address", "Unknown Address"),
                        "store_latitude": store_lat,
                        "store_longitude": store_lng,
                        "distance_km": round(distance, 2),
                        "recommended_perfumes": matched_perfumes
                    })
                    logger.info(
                        f"[COURSE] 매장 추가: {store.get('name')} (거리: {distance:.2f}km, 향수: {len(matched_perfumes)}개)")

        logger.info(f"[COURSE] 반경 {radius_km}km 내 추천 가능한 매장: {len(nearby_stores)}개")

        # ─── 7. 응답 데이터 구성 ───────────────────────────────────────────────────────
        if not nearby_stores:
            logger.info("[COURSE] 추천 가능한 매장이 없음")
            return CourseRecommendResponse(
                message="주변에 추천 향수를 판매하는 매장이 없습니다. 검색 반경을 늘려보세요.",
                total_stores=0,
                recommended_perfumes_count=0,
                data=[]
            )

        # 거리순으로 정렬
        nearby_stores.sort(key=lambda x: x["distance_km"])

        # CourseItem 객체로 변환
        course_items = []
        total_perfumes = 0
        for store in nearby_stores:
            course_item = CourseItem(
                store_name=store["store_name"],
                store_address=store["store_address"],
                store_latitude=store["store_latitude"],
                store_longitude=store["store_longitude"],
                distance_km=store["distance_km"],
                recommended_perfumes=store["recommended_perfumes"]
            )
            course_items.append(course_item)
            total_perfumes += len(store["recommended_perfumes"])

        logger.info(f"[COURSE] 최종 응답: {len(course_items)}개 매장, {total_perfumes}개 추천 향수")

        return CourseRecommendResponse(
            message=f"AI 모델 기반 향수 코스 추천이 완료되었습니다. 반경 {radius_km}km 내 {len(course_items)}개 매장에서 {total_perfumes}개의 향수를 확인하실 수 있습니다.",
            total_stores=len(course_items),
            recommended_perfumes_count=total_perfumes,
            data=course_items
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[COURSE] 예상치 못한 오류: {e}")
        raise HTTPException(status_code=500, detail=f"코스 추천 중 오류가 발생했습니다: {str(e)}")