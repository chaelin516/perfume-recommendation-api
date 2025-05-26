from fastapi import APIRouter
from fastapi.responses import JSONResponse
from schemas.course import SimpleCourseRecommendRequest
import pandas as pd
import json, math, os

router = APIRouter(prefix="/courses", tags=["Course"])

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
store_path = os.path.join(BASE_DIR, "../data/store_data.json")
perfume_path = os.path.join(BASE_DIR, "../data/perfume_final_dataset.csv")

# 데이터 로딩
with open(store_path, "r", encoding="utf-8") as f:
    store_data = json.load(f)

perfume_df = pd.read_csv(perfume_path)
perfume_df.fillna("", inplace=True)


# 거리 계산 함수 (Haversine 공식)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# 리스트 or 문자열 비교
def is_match(field, value):
    if isinstance(field, list):
        return value in field
    return value == field

# CSV 안에서 emotion/season/time 태그 파싱 비교
def tag_match(cell: str, tag: str) -> bool:
    return tag in cell.split(", ")


# 향수 코스 추천 API
@router.post(
    "/recommend",
    summary="향수 코스 추천",
    description="성별, 감정, 계절, 시간 및 위치를 기반으로 추천 향수와 매장을 반환",
    response_description="추천된 향수 및 매장 목록을 거리 순으로 정렬하여 반환"
)
async def recommend_course(request: SimpleCourseRecommendRequest):
    try:
        # 조건 필터링
        matched_perfumes = perfume_df[
            (perfume_df["gender"] == request.gender) &
            (perfume_df["emotion_tags"].apply(lambda x: tag_match(x, request.emotion))) &
            (perfume_df["season_tags"].apply(lambda x: tag_match(x, request.season))) &
            (perfume_df["time_tags"].apply(lambda x: tag_match(x, request.time)))
        ]

        course_list = []

        for _, perfume in matched_perfumes.iterrows():
            perfume_name = perfume["name"]
            for store in store_data:
                if perfume_name in store.get("perfumes", []):
                    distance = haversine(
                        request.latitude,
                        request.longitude,
                        store.get("lat", 0.0),
                        store.get("lng", 0.0)
                    )
                    course_list.append({
                        "store": store.get("name", ""),
                        "address": store.get("address", ""),
                        "perfume_name": perfume_name,
                        "brand": perfume["brand"],
                        "image_url": perfume["image_url"],
                        "distance_km": round(distance, 2)
                    })

        sorted_course = sorted(course_list, key=lambda x: x["distance_km"])[:5]

        return JSONResponse(
            status_code=200,
            content={
                "message": "향수 코스 추천 성공",
                "data": sorted_course
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "message": f"향수 코스 추천 실패: {str(e)}",
                "data": None
            }
        )
