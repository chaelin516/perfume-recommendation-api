from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from perfume_backend.schemas.base import BaseResponse
import pandas as pd
import os

router = APIRouter(prefix="/perfumes", tags=["Perfume"])

# CSV 파일 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
perfume_path = os.path.join(BASE_DIR, "../data/perfume_final_dataset.csv")

# CSV 데이터 로딩
df = pd.read_csv(perfume_path)
df["emotion_tags"] = df["emotion_tags"].fillna("").astype(str)
df["notes"] = df["notes"].fillna("").astype(str)
df["image_url"] = df["image_url"].fillna("").astype(str)


# ✅ 전체 향수 목록 조회
@router.get(
    "/",
    response_model=BaseResponse,
    summary="전체 향수 목록 조회",
    description="저장된 전체 향수 데이터를 리스트로 반환합니다.",
    response_description="향수 목록 리스트 반환"
)
async def get_all_perfumes():
    perfumes = df[["name", "brand", "image_url"]].to_dict(orient="records")
    return BaseResponse(
        code=200,
        message="전체 향수 목록입니다.",
        data={"perfumes": perfumes}
    )


# ✅ 특정 향수 상세 조회
@router.get(
    "/{name}",
    response_model=BaseResponse,
    summary="향수 상세 정보 조회",
    description="지정한 이름에 해당하는 향수의 상세 정보를 반환합니다.",
    response_description="향수 상세 데이터 반환"
)
async def get_perfume_detail(name: str):
    match = df[df["name"] == name]
    if match.empty:
        raise HTTPException(status_code=404, detail="해당 이름의 향수를 찾을 수 없습니다.")

    row = match.iloc[0]
    result = {
        "name": row["name"],
        "brand": row["brand"],
        "image_url": row["image_url"],
        "notes": row["notes"],
        "emotion_tags": row["emotion_tags"]
    }

    return BaseResponse(
        code=200,
        message=f"{name} 향수의 상세 정보입니다.",
        data=result
    )
