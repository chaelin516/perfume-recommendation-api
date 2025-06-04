# project_root/routers/recommend_router.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Literal
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import load_model
import os

# ─── 1. 데이터(향수 메타) 로드 ───────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/perfume_final_dataset.csv")
df = pd.read_csv(DATA_PATH)
df.fillna("", inplace=True)

# ─── 2. Keras 모델 경로 지정 ────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/final_model_perfume.keras")

# ─── 3. 전역 변수(lazy loading) ─────────────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
    return _model

# ─── 4. OneHotEncoder 준비: (gender, season, time, impression, activity, weather) 순서 ────
#    - categories 리스트 안에 “허용 가능한 문자열”을 반드시 모두 나열해야 함
#    - handle_unknown="ignore"로 지정하면, transform 시 허용되지 않은 값이 들어와도 에러 대신 0 벡터로 처리
CATEGORIES = [
    ["women", "men", "unisex"],                     # gender
    ["spring", "summer", "fall", "winter"],          # season
    ["day", "night"],                                # time
    ["confident", "elegant", "pure", "friendly", "mysterious", "fresh"],  # impression
    ["casual", "work", "date"],                      # activity
    ["hot", "cold", "rainy", "any"]                  # weather
]
_encoder = OneHotEncoder(categories=CATEGORIES, handle_unknown="ignore", sparse=False)

# ※ 이미 categories를 직접 지정했기 때문에, 별도의 fit() 호출 없이 곧바로 transform()만 사용 가능합니다.

# ─── 5. 요청(Request) 스키마 정의 ───────────────────────────────────────────────
class RecommendRequest(BaseModel):
    gender: Literal["women", "men", "unisex"]
    season: Literal["spring", "summer", "fall", "winter"]
    time: Literal["day", "night"]
    impression: Literal["confident", "elegant", "pure", "friendly", "mysterious", "fresh"]
    activity: Literal["casual", "work", "date"]
    weather: Literal["hot", "cold", "rainy", "any"]

# ─── 6. 응답(Response) 스키마 정의 ───────────────────────────────────────────────
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
        "OneHotEncoder → Keras 모델에 그대로 전달하여, 예측 점수가 높은 상위 10개 향수를 반환합니다."
    )
)
def recommend_perfumes(request: RecommendRequest):
    # 1) 클라이언트 요청에서 받은 6개 문자열을 그대로 raw_features에 담는다
    #    → 순서: [gender, season, time, impression, activity, weather]
    raw_features = [
        request.gender,
        request.season,
        request.time,
        request.impression,
        request.activity,
        request.weather
    ]

    # 2) OneHotEncoder.transform() 으로 숫자형 입력 행렬(x_input) 생성
    try:
        # transform 메서드는 2D 배열을 기대하므로, [raw_features]처럼 리스트 안에 리스트 형태로 넘김
        x_input = _encoder.transform([raw_features])  # 결과는 NumPy 배열(shape: (1, 총 원-핫 차원))
    except Exception as e:
        # 만약 raw_features 값이 CATEGORIES에 정의되지 않았다면 이 부분에서 예외 발생
        raise HTTPException(status_code=400, detail=f"입력값 전처리 중 오류: {str(e)}")

    # 3) 모델 불러와서 예측(predict)
    model = get_model()
    try:
        preds = model.predict(x_input)  # preds.shape == (1, num_perfumes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 예측 중 오류: {str(e)}")

    # 4) 1차원 배열로 변환
    scores = preds.flatten()  # shape: (num_perfumes,)

    # 5) 모델 출력 크기(num_perfumes)와 DataFrame 행 수(df.shape[0]) 비교
    if len(scores) != len(df):
        raise HTTPException(
            status_code=500,
            detail="모델 출력 크기가 데이터프레임 행 개수와 맞지 않습니다."
        )

    # 6) 원본 df 복사본에 “score” 컬럼을 추가
    df_with_scores = df.copy()
    df_with_scores["score"] = scores

    # 7) score 기준 내림차순 정렬 후 상위 10개(Head 10)만 추출
    top_10 = df_with_scores.sort_values(by="score", ascending=False).head(10)

    # 8) 최종 응답 리스트 생성
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
