import os
import pickle
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Literal
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import load_model

# ─── 로거 설정 ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("recommend_router")

# ─── 1. perfume_final_dataset.csv 로드 ───────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/perfume_final_dataset.csv")
try:
    df = pd.read_csv(DATA_PATH)
    df.fillna("", inplace=True)
    logger.info(f"Perfume dataset loaded: {df.shape[0]} rows")
except Exception as e:
    logger.error(f"perfume_final_dataset.csv 로드 중 오류: {e}")
    raise RuntimeError(f"perfume_final_dataset.csv 로드 중 오류: {e}")

# ─── 2. Keras 모델 및 encoder.pkl 파일 경로 설정 ─────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/final_model_perfume.keras")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "../models/encoder.pkl")

# ─── 3. 전역 변수(lazy loading) 및 OneHotEncoder 설정 ─────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        try:
            _model = load_model(MODEL_PATH)
            logger.info("Keras 모델 로드 성공")
        except Exception as e:
            logger.error(f"Keras 모델 로드 중 오류: {e}")
            raise RuntimeError(f"Keras 모델 로드 중 오류: {e}")
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
        raise RuntimeError(f"encoder.pkl 로드 중 오류: {e}")

# fallback OneHotEncoder: 카테고리 직접 선언, handle_unknown="ignore" 설정
CATEGORIES = [
    ["women", "men", "unisex"],                     # gender
    ["spring", "summer", "fall", "winter"],          # season
    ["day", "night"],                                # time
    ["confident", "elegant", "pure", "friendly", "mysterious", "fresh"],  # impression
    ["casual", "work", "date"],                      # activity
    ["hot", "cold", "rainy", "any"]                  # weather
]
_fallback_encoder = OneHotEncoder(categories=CATEGORIES, handle_unknown="ignore", sparse=False)
# 더미 데이터를 이용해 한 번 fit() 호출
_fallback_encoder.fit([
    ["women", "spring", "day", "confident", "casual", "hot"],
    ["men", "summer", "night", "elegant", "work", "cold"]
])

# ─── 4. 요청(Request) 스키마 정의 ────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    gender: Literal["women", "men", "unisex"]
    season: Literal["spring", "summer", "fall", "winter"]
    time: Literal["day", "night"]
    impression: Literal["confident", "elegant", "pure", "friendly", "mysterious", "fresh"]
    activity: Literal["casual", "work", "date"]
    weather: Literal["hot", "cold", "rainy", "any"]

# ─── 5. 응답(Response) 스키마 정의 ────────────────────────────────────────────────
class PerfumeRecommendItem(BaseModel):
    name: str
    brand: str
    image_url: str
    notes: str
    emotions: str
    reason: str

router = APIRouter(prefix="/perfumes", tags=["Perfume"])

@router.post(
    "/recommend",
    response_model=List[PerfumeRecommendItem],
    summary="모델 기반 향수 추천",
    description=(
        "gender, season, time, impression, activity, weather 정보를 입력받아\n"
        "encoder.pkl 또는 fallback OneHotEncoder로 전처리 → Keras 모델 예측으로\n"
        "향수 10개를 추천합니다."
    )
)
def recommend_perfumes(request: RecommendRequest):
    logger.info(f"[PERFUME] 요청 파라미터: {request}")

    # 1) raw_features: 요청 값 6개를 순서대로 리스트로 묶기
    raw_features = [
        request.gender,
        request.season,
        request.time,
        request.impression,
        request.activity,
        request.weather
    ]

    # 2) encoder.pkl 사용 시도
    use_fallback = False
    try:
        encoder = get_saved_encoder()
        x_input = encoder.transform([raw_features])  # (1, D_saved) 형태
        logger.info("[PERFUME] encoder.pkl 사용하여 전처리 성공")
    except Exception as e:
        logger.warning(f"[PERFUME] encoder.pkl 전처리 실패: {e} → fallback 사용")
        use_fallback = True

    # 3) fallback OneHotEncoder 사용 (encoder.pkl 실패 시)
    if use_fallback:
        try:
            x_input = _fallback_encoder.transform([raw_features])  # (1, D_fallback) 형태
            logger.info("[PERFUME] fallback OneHotEncoder 사용하여 전처리 성공")
        except Exception as e:
            logger.error(f"[PERFUME] fallback 전처리 중 오류: {e}")
            raise HTTPException(status_code=400, detail=f"입력값 전처리 중 오류: {e}")

    # 4) Keras 모델 예측
    try:
        model = get_model()
        preds = model.predict(x_input)  # preds.shape == (1, num_perfumes)
        logger.info("[PERFUME] 모델 예측 성공")
    except Exception as e:
        logger.error(f"[PERFUME] 모델 예측 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"모델 예측 중 오류: {e}")

    # 5) 1차원 배열로 변환
    scores = preds.flatten()  # (num_perfumes,) 형태

    # 6) 모델 출력 크기와 DataFrame 행 수 검증
    if len(scores) != len(df):
        logger.error("[PERFUME] 모델 출력 크기와 데이터 행 개수 불일치")
        raise HTTPException(
            status_code=500,
            detail="모델 출력 크기가 perfume_final_dataset.csv의 행 개수와 일치하지 않습니다."
        )

    # 7) DataFrame 복사본에 'score' 컬럼 추가
    df_with_scores = df.copy()
    df_with_scores["score"] = scores

    # 8) score 순으로 내림차순 정렬 후 상위 10개 추출
    top_10 = df_with_scores.sort_values(by="score", ascending=False).head(10)
    logger.info(f"[PERFUME] 상위 추천 향수 10개 선정 완료")

    # 9) 결과 가공하여 반환
    response_list: List[PerfumeRecommendItem] = []
    for _, row in top_10.iterrows():
        response_list.append(
            PerfumeRecommendItem(
                name=row["name"],
                brand=row["brand"],
                image_url=row["image_url"],
                notes=row["notes"],
                emotions=row["emotion_tags"],
                reason=f"추천 이유: 모델 예측 점수 {row['score']:.3f}"
            )
        )

    return response_list
