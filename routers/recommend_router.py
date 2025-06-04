# project_root/routers/recommend_router.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Literal
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import load_model
import pickle
import os

# ─── 1. perfume_final_dataset.csv 로드 ───────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/perfume_final_dataset.csv")
try:
    df = pd.read_csv(DATA_PATH)
    df.fillna("", inplace=True)
except Exception as e:
    raise RuntimeError(f"perfume_final_dataset.csv 로드 중 오류: {e}")

# ─── 2. Keras 모델 및 encoder.pkl 파일 경로 설정 ─────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/final_model_perfume.keras")
ENCODER_PATH = os.path.join(os.path.dirname(__file__), "../models/encoder.pkl")

# ─── 3. 전역 변수(lazy loading) ───────────────────────────────────────────────────────
_model = None
_saved_encoder = None

def get_model():
    global _model
    if _model is None:
        try:
            _model = load_model(MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"Keras 모델 로드 중 오류: {e}")
    return _model

def load_saved_encoder():
    global _saved_encoder
    if _saved_encoder is None:
        try:
            with open(ENCODER_PATH, "rb") as f:
                _saved_encoder = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"encoder.pkl 로드 중 오류: {e}")
    return _saved_encoder

# ─── 4. OneHotEncoder 직접 생성 (fallback) ─────────────────────────────────────────────
#    만약 saved_encoder 로드 후 transform 시 “unknown categories” 오류가 난다면,
#    이 직접 생성한 encoder를 사용하도록 합니다.
CATEGORIES = [
    ["women", "men", "unisex"],                     # gender
    ["spring", "summer", "fall", "winter"],          # season
    ["day", "night"],                                # time
    ["confident", "elegant", "pure", "friendly", "mysterious", "fresh"],  # impression
    ["casual", "work", "date"],                      # activity
    ["hot", "cold", "rainy", "any"]                  # weather
]
_fallback_encoder = OneHotEncoder(categories=CATEGORIES, handle_unknown="ignore", sparse=False)
# fallback_encoder는 일반적으로 한 번 fit() 호출 후 transform() 사용
_fallback_encoder.fit(
    [
        ["women", "spring", "day", "confident", "casual", "hot"],
        ["men", "summer", "night", "elegant", "work", "cold"]
    ]
)

# ─── 5. 요청(Request) 스키마 정의 ────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    gender: Literal["women", "men", "unisex"]
    season: Literal["spring", "summer", "fall", "winter"]
    time: Literal["day", "night"]
    impression: Literal["confident", "elegant", "pure", "friendly", "mysterious", "fresh"]
    activity: Literal["casual", "work", "date"]
    weather: Literal["hot", "cold", "rainy", "any"]

# ─── 6. 응답(Response) 스키마 정의 ────────────────────────────────────────────────────
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
        "API로 받은 gender, season, time, impression, activity, weather 정보를 "
        "encoder.pkl 또는 fallback OneHotEncoder로 전처리 → Keras 모델에 전달하여, "
        "예측 점수가 높은 상위 10개 향수를 반환합니다."
    )
)
def recommend_perfumes(request: RecommendRequest):
    # ─── 1) 클라이언트 요청 데이터(raw_features) 준비 ───────────────────────────────
    raw_features = [
        request.gender,
        request.season,
        request.time,
        request.impression,
        request.activity,
        request.weather
    ]

    # ─── 2) encoder.pkl 사용 시도 ───────────────────────────────────────────────────
    use_fallback = False
    try:
        encoder = load_saved_encoder()
        # encoder.transform이 2D 배열을 기대하므로 리스트 안에 리스트 형태로 전달
        x_input = encoder.transform([raw_features])  # (1, D_saved) 예상
    except Exception as e:
        # encoder.pkl 변환이 실패하면 fallback 사용
        use_fallback = True

    # ─── 3) fallback_encoder 사용(encoder.pkl 실패 시) ──────────────────────────────
    if use_fallback:
        try:
            x_input = _fallback_encoder.transform([raw_features])  # (1, D_fallback)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"입력값 전처리 중 오류(2차 시도): {e}")

    # ─── 4) Keras 모델 예측(predict) ────────────────────────────────────────────────
    try:
        model = get_model()
        preds = model.predict(x_input)  # preds.shape == (1, num_perfumes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 예측 중 오류: {e}")

    # ─── 5) 1차원 배열로 변환 ────────────────────────────────────────────────────────
    scores = preds.flatten()  # (num_perfumes,) 형태

    # ─── 6) 모델 출력 개수와 DataFrame 행 수 일치 검증 ───────────────────────────────
    if len(scores) != len(df):
        raise HTTPException(
            status_code=500,
            detail="모델 출력 크기가 perfume_final_dataset.csv의 행 개수와 일치하지 않습니다."
        )

    # ─── 7) DataFrame 복사본에 점수(score) 컬럼 추가 ─────────────────────────────────
    df_with_scores = df.copy()
    df_with_scores["score"] = scores

    # ─── 8) score 기준 내림차순 정렬 후 상위 10개 추출 ─────────────────────────────────
    top_10 = df_with_scores.sort_values(by="score", ascending=False).head(10)

    # ─── 9) 최종 응답 리스트 생성 ────────────────────────────────────────────────────
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
