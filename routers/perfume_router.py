# routers/perfume_router.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from schemas.base import BaseResponse  # ✅ 절대경로 import
import pandas as pd
import os
import logging

router = APIRouter(prefix="/perfumes", tags=["Perfume"])

# 로거 설정
logger = logging.getLogger("perfume_router")

# CSV 파일 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
perfume_path = os.path.join(BASE_DIR, "../data/perfume_final_dataset.csv")

# ✅ CSV 데이터 로딩 - 실제 컬럼 구조에 맞춤
try:
    df = pd.read_csv(perfume_path)
    logger.info(f"✅ Perfume CSV 로드 성공: {df.shape[0]}행 x {df.shape[1]}열")

    # 🔍 실제 컬럼 정보 로그
    logger.info("📋 실제 CSV 컬럼 목록:")
    for i, col in enumerate(df.columns, 1):
        logger.info(f"  {i:2d}. '{col}'")

    # ✅ 실제 존재하는 컬럼들로 안전한 전처리
    expected_columns = [
        'name', 'brand', 'image_url', 'gender', 'notes',
        'season_tags', 'time_tags', 'brand_tag', 'activity',
        'weather', 'desired_impression', 'emotion_cluster'
    ]

    for col in expected_columns:
        if col in df.columns:
            if col == 'emotion_cluster':
                # 정수형 컬럼은 NaN을 0으로 처리
                df[col] = df[col].fillna(0).astype(int)
                logger.info(f"✅ '{col}' 컬럼 정수형 전처리 완료")
            else:
                # 문자열 컬럼은 빈 문자열로 처리
                df[col] = df[col].fillna("").astype(str)
                logger.info(f"✅ '{col}' 컬럼 문자열 전처리 완료")
        else:
            logger.warning(f"⚠️ '{col}' 컬럼이 CSV에 없습니다.")

    # 🔍 데이터 샘플 로그
    if len(df) > 0:
        logger.info("📝 첫 번째 행 샘플:")
        sample_row = df.iloc[0]
        for col in df.columns:
            logger.info(f"  {col}: '{sample_row[col]}'")

        # 🔍 주요 컬럼들의 고유값 확인
        logger.info("🔍 주요 컬럼 고유값 정보:")

        # 성별 고유값
        if 'gender' in df.columns:
            genders = df['gender'].unique()
            logger.info(f"  성별 고유값: {genders.tolist()}")

        # 계절 태그에서 개별 태그 추출
        if 'season_tags' in df.columns:
            all_seasons = set()
            for _, row in df.head(100).iterrows():  # 처음 100개만 확인
                if pd.notna(row['season_tags']) and str(row['season_tags']).strip():
                    seasons = [s.strip() for s in str(row['season_tags']).split(',')]
                    all_seasons.update([s for s in seasons if s])
            logger.info(f"  계절 태그 예시: {sorted(list(all_seasons))}")

        # 시간 태그에서 개별 태그 추출
        if 'time_tags' in df.columns:
            all_times = set()
            for _, row in df.head(100).iterrows():  # 처음 100개만 확인
                if pd.notna(row['time_tags']) and str(row['time_tags']).strip():
                    times = [t.strip() for t in str(row['time_tags']).split(',')]
                    all_times.update([t for t in times if t])
            logger.info(f"  시간 태그 예시: {sorted(list(all_times))}")

        # 원하는 인상 고유값
        if 'desired_impression' in df.columns:
            impressions = df['desired_impression'].unique()
            logger.info(f"  원하는 인상 고유값: {impressions.tolist()}")

        # 활동 고유값
        if 'activity' in df.columns:
            activities = df['activity'].unique()
            logger.info(f"  활동 고유값: {activities.tolist()}")

except FileNotFoundError:
    logger.error(f"❌ CSV 파일을 찾을 수 없습니다: {perfume_path}")
    df = pd.DataFrame()
except Exception as e:
    logger.error(f"❌ CSV 로딩 중 오류: {e}")
    df = pd.DataFrame()


# ✅ 전체 향수 목록 조회
@router.get(
    "/",
    response_model=BaseResponse,
    summary="전체 향수 목록 조회",
    description="저장된 전체 향수 데이터를 리스트로 반환합니다.",
    response_description="향수 목록 리스트 반환"
)
async def get_all_perfumes():
    try:
        if df.empty:
            return BaseResponse(
                code=404,
                message="향수 데이터를 찾을 수 없습니다.",
                data={"perfumes": []}
            )

        # 기본 필드들만 선택
        basic_columns = ['name', 'brand', 'image_url']
        available_columns = [col for col in basic_columns if col in df.columns]

        if not available_columns:
            # 사용 가능한 컬럼이 없으면 처음 3개 컬럼 사용
            available_columns = df.columns[:3].tolist()
            logger.warning(f"⚠️ 기본 컬럼을 찾을 수 없어 처음 3개 컬럼 사용: {available_columns}")

        perfumes = df[available_columns].to_dict(orient="records")

        logger.info(f"✅ 향수 목록 조회 완료: {len(perfumes)}개")

        return BaseResponse(
            code=200,
            message="전체 향수 목록입니다.",
            data={"perfumes": perfumes}
        )

    except Exception as e:
        logger.error(f"❌ 향수 목록 조회 중 오류: {e}")
        return BaseResponse(
            code=500,
            message=f"서버 오류가 발생했습니다: {str(e)}",
            data={"perfumes": []}
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
    try:
        if df.empty:
            raise HTTPException(status_code=404, detail="향수 데이터를 찾을 수 없습니다.")

        # 이름으로 향수 찾기
        if 'name' not in df.columns:
            raise HTTPException(status_code=500, detail="name 컬럼이 데이터에 없습니다.")

        match = df[df['name'] == name]
        if match.empty:
            raise HTTPException(status_code=404, detail="해당 이름의 향수를 찾을 수 없습니다.")

        row = match.iloc[0]

        # ✅ 실제 컬럼 구조에 맞춘 결과 구성
        result = {}

        # 기본 정보
        result['name'] = row.get('name', '')
        result['brand'] = row.get('brand', '')
        result['image_url'] = row.get('image_url', '')
        result['notes'] = row.get('notes', '')

        # 태그 정보
        result['season_tags'] = row.get('season_tags', '')
        result['time_tags'] = row.get('time_tags', '')
        result['brand_tag'] = row.get('brand_tag', '')

        # 새로운 필드들
        result['activity'] = row.get('activity', '')
        result['weather'] = row.get('weather', '')
        result['desired_impression'] = row.get('desired_impression', '')
        result['emotion_cluster'] = int(row.get('emotion_cluster', 0))
        result['gender'] = row.get('gender', '')

        logger.info(f"✅ 향수 상세 조회 완료: {name}")

        return BaseResponse(
            code=200,
            message=f"{name} 향수의 상세 정보입니다.",
            data=result
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 향수 상세 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"서버 오류가 발생했습니다: {str(e)}")


# 🔍 CSV 구조 확인을 위한 디버그 엔드포인트
@router.get(
    "/debug/csv-info",
    summary="CSV 파일 구조 정보 (디버그용)",
    description="CSV 파일의 구조와 컬럼 정보를 반환합니다."
)
async def get_csv_debug_info():
    try:
        if df.empty:
            return {
                "message": "CSV 파일이 로드되지 않았습니다.",
                "file_path": perfume_path,
                "error": "DataFrame이 비어있습니다."
            }

        # 📊 기본 정보
        info = {
            "file_path": perfume_path,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "memory_usage": df.memory_usage(deep=True).to_dict()
        }

        # 🔍 각 컬럼의 상세 정보
        column_details = {}
        for col in df.columns:
            column_details[col] = {
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "null_percentage": round((df[col].isnull().sum() / len(df)) * 100, 2),
                "unique_count": int(df[col].nunique()),
                "sample_values": df[col].dropna().head(3).tolist() if not df[col].dropna().empty else []
            }

        info["column_details"] = column_details

        # 📝 샘플 데이터 (처음 3행)
        if len(df) > 0:
            info["sample_data"] = df.head(3).to_dict(orient="records")

        return info

    except Exception as e:
        logger.error(f"❌ CSV 디버그 정보 조회 중 오류: {e}")
        return {
            "error": f"디버그 정보 조회 중 오류: {str(e)}",
            "file_path": perfume_path
        }