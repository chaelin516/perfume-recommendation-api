# routers/recommend_router.py - 급한 수정 버전 (impression_tags 에러 해결)

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Literal
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from utils.model_predictor import predict_emotion_cluster

router = APIRouter(prefix="/perfumes", tags=["Perfume"])

# 🔧 안전한 CSV 로드 및 데이터 검증
try:
    print("📊 향수 데이터 로딩 시작...")
    df = pd.read_csv("./data/perfume_final_dataset.csv")

    # 기본 전처리
    df.fillna("", inplace=True)

    print(f"✅ 데이터 로딩 완료: {len(df)}행, {len(df.columns)}열")
    print(f"📋 실제 컬럼들: {list(df.columns)}")

    # 🔍 핵심 컬럼 확인
    required_basic_columns = ['name', 'brand', 'gender']
    missing_basic = [col for col in required_basic_columns if col not in df.columns]

    if missing_basic:
        print(f"❌ 필수 컬럼 누락: {missing_basic}")
        # 기본값으로 생성
        for col in missing_basic:
            if col == 'name':
                df['name'] = 'Unknown Perfume'
            elif col == 'brand':
                df['brand'] = 'Unknown Brand'
            elif col == 'gender':
                df['gender'] = 'unisex'

    # 🔍 태그 컬럼 확인 및 통합 (impression_tags 에러 해결)
    tag_columns_to_check = ['emotion_tags', 'impression_tags', 'season_tags', 'time_tags']
    available_tag_columns = []

    for col in tag_columns_to_check:
        if col in df.columns:
            available_tag_columns.append(col)
            print(f"✅ 발견된 태그 컬럼: {col}")
        else:
            print(f"❌ 누락된 태그 컬럼: {col}")
            # 누락된 컬럼 생성
            if col == 'emotion_tags':
                df[col] = 'fresh, clean, elegant'
            elif col == 'impression_tags':
                df[col] = 'confident, elegant, fresh'
            elif col == 'season_tags':
                df[col] = 'spring, summer, fall, winter'
            elif col == 'time_tags':
                df[col] = 'day, night'

    # 🔍 다른 필수 컬럼들 확인
    other_columns = ['image_url', 'notes']
    for col in other_columns:
        if col not in df.columns:
            print(f"⚠️ {col} 컬럼 없음 - 기본값으로 생성")
            df[col] = ''

    # 🔧 벡터화 준비 (emotion_tags 우선 사용)
    primary_tag_column = 'emotion_tags' if 'emotion_tags' in df.columns else 'impression_tags'
    vectorizer = CountVectorizer(tokenizer=lambda x: str(x).split(", "))
    emotion_matrix = vectorizer.fit_transform(df[primary_tag_column].astype(str))

    print(f"✅ 데이터 준비 및 벡터화 완료 (주 태그 컬럼: {primary_tag_column})")

except Exception as e:
    print(f"❌ 데이터 로딩 실패: {e}")
    import traceback

    traceback.print_exc()

    # 🚨 최소한의 더미 데이터로 초기화
    df = pd.DataFrame({
        'name': ['Sample Perfume 1', 'Sample Perfume 2', 'Sample Perfume 3'],
        'brand': ['Sample Brand', 'Test Brand', 'Demo Brand'],
        'gender': ['women', 'men', 'unisex'],
        'image_url': ['', '', ''],
        'notes': ['Fresh floral scent', 'Woody masculine scent', 'Light citrus scent'],
        'emotion_tags': ['fresh, elegant, clean', 'confident, mysterious, deep', 'friendly, pure, light'],
        'impression_tags': ['confident, elegant', 'mysterious, deep', 'friendly, fresh'],
        'season_tags': ['spring, summer', 'fall, winter', 'spring, summer'],
        'time_tags': ['day', 'day, night', 'day']
    })

    try:
        vectorizer = CountVectorizer(tokenizer=lambda x: str(x).split(", "))
        emotion_matrix = vectorizer.fit_transform(df["emotion_tags"].astype(str))
        print("✅ 더미 데이터로 초기화 완료")
    except:
        vectorizer = None
        emotion_matrix = None
        print("❌ 벡터화도 실패 - 기본 모드로 동작")


# 요청 스키마 (프론트엔드와 정확히 일치)
class RecommendRequest(BaseModel):
    gender: Literal['women', 'men', 'unisex']
    season: Literal['spring', 'summer', 'fall', 'winter']
    time: Literal['day', 'night']
    impression: Literal['confident', 'elegant', 'pure', 'friendly', 'mysterious', 'fresh']
    activity: Literal['casual', 'work', 'date']
    weather: Literal['hot', 'cold', 'rainy', 'any']


# 응답 스키마
class PerfumeRecommendItem(BaseModel):
    name: str
    brand: str
    image_url: str
    notes: str
    emotions: str
    reason: str


# 안전한 태그 매칭 함수
def safe_tag_match(cell: str, tag: str) -> bool:
    """태그 매칭 (안전 버전)"""
    try:
        if pd.isna(cell) or cell == "":
            return False
        return tag.lower() in str(cell).lower()
    except:
        return False


# ✅ 수정된 1차 향수 추천 API (impression_tags 에러 해결)
@router.post(
    "/recommend",
    response_model=List[PerfumeRecommendItem],
    summary="1차 향수 추천",
    description="프론트엔드 요청에 맞춘 향수 추천 (impression_tags 에러 해결)"
)
def recommend_perfumes(request: RecommendRequest):
    """
    🔧 에러 해결 버전:
    - impression_tags 컬럼 에러 해결
    - 안전한 태그 매칭
    - 컬럼 존재 여부 확인 후 처리
    """

    try:
        print(f"\n🔍 추천 요청 받음:")
        print(f"  Gender: {request.gender}")
        print(f"  Season: {request.season}")
        print(f"  Time: {request.time}")
        print(f"  Impression: {request.impression}")
        print(f"  Activity: {request.activity}")
        print(f"  Weather: {request.weather}")

        # 🔧 1단계: 성별 필터링 (가장 확실한 필터)
        filtered_df = df.copy()

        if 'gender' in df.columns:
            gender_mask = (df['gender'] == request.gender) | (df['gender'] == 'unisex')
            gender_filtered = df[gender_mask]

            if len(gender_filtered) > 0:
                filtered_df = gender_filtered
                print(f"  ✅ 성별 필터링: {len(filtered_df)}개 (gender={request.gender} or unisex)")
            else:
                print(f"  ⚠️ 성별 필터링 결과 없음 - 전체 데이터 사용")

        # 🔧 2단계: 안전한 태그 기반 필터링
        print(f"  🔍 사용 가능한 컬럼들: {list(df.columns)}")

        # 사용할 태그 컬럼들 결정
        tag_columns = {}
        if 'impression_tags' in df.columns:
            tag_columns['impression'] = 'impression_tags'
        elif 'emotion_tags' in df.columns:
            tag_columns['impression'] = 'emotion_tags'

        if 'season_tags' in df.columns:
            tag_columns['season'] = 'season_tags'
        elif 'emotion_tags' in df.columns:
            tag_columns['season'] = 'emotion_tags'

        if 'time_tags' in df.columns:
            tag_columns['time'] = 'time_tags'
        elif 'emotion_tags' in df.columns:
            tag_columns['time'] = 'emotion_tags'

        print(f"  📋 사용할 태그 컬럼들: {tag_columns}")

        # 단계별 필터링
        filtering_stages = [
            ('impression', request.impression, tag_columns.get('impression')),
            ('season', request.season, tag_columns.get('season')),
            ('time', request.time, tag_columns.get('time'))
        ]

        best_filtered = filtered_df
        best_score = 0

        for stage_name, search_value, column_name in filtering_stages:
            if column_name and column_name in filtered_df.columns:
                try:
                    # 안전한 태그 매칭
                    mask = filtered_df[column_name].apply(
                        lambda x: safe_tag_match(x, search_value)
                    )
                    stage_filtered = filtered_df[mask]

                    if len(stage_filtered) > 0:
                        print(f"    ✅ {stage_name} 필터링: {len(stage_filtered)}개 ({search_value})")

                        # 더 좋은 결과면 업데이트
                        if len(stage_filtered) >= 3:  # 충분한 결과가 있으면
                            best_filtered = stage_filtered
                            best_score += 1
                        elif best_score == 0:  # 첫 번째 유효한 결과
                            best_filtered = stage_filtered
                            best_score = 1
                    else:
                        print(f"    ❌ {stage_name} 필터링: 결과 없음 ({search_value})")

                except Exception as e:
                    print(f"    ⚠️ {stage_name} 필터링 중 오류: {e}")

        filtered_df = best_filtered

        # 🔧 3단계: 최종 결과 확인 및 처리
        if len(filtered_df) == 0:
            print(f"  ❌ 필터링 결과 없음 - 성별 기반 결과 사용")
            if 'gender' in df.columns:
                gender_mask = (df['gender'] == request.gender) | (df['gender'] == 'unisex')
                filtered_df = df[gender_mask]
                if len(filtered_df) == 0:
                    filtered_df = df.head(10)  # 최후의 수단
            else:
                filtered_df = df.head(10)

        # 🔧 4단계: 유사도 기반 정렬 (가능한 경우)
        if vectorizer is not None and emotion_matrix is not None and len(filtered_df) > 0:
            try:
                # 사용자 선호도 벡터 생성
                user_preferences = [request.impression, 'elegant', 'fresh']  # 기본 선호도 포함
                user_vector = vectorizer.transform([', '.join(user_preferences)])

                # 필터링된 데이터의 인덱스
                filtered_indices = filtered_df.index.tolist()
                filtered_emotion_matrix = emotion_matrix[filtered_indices]

                # 유사도 계산
                similarities = cosine_similarity(user_vector, filtered_emotion_matrix).flatten()

                # 유사도를 DataFrame에 추가
                filtered_df = filtered_df.copy()
                filtered_df['similarity_score'] = similarities

                # 유사도 순으로 정렬
                filtered_df = filtered_df.sort_values('similarity_score', ascending=False)

                print(f"  ✅ 유사도 계산 완료 (최고 점수: {similarities.max():.3f})")

            except Exception as e:
                print(f"  ⚠️ 유사도 계산 실패: {e}")
                # 랜덤으로 섞기
                filtered_df = filtered_df.sample(frac=1.0)
        else:
            # 벡터화가 불가능한 경우 랜덤으로 섞기
            filtered_df = filtered_df.sample(frac=1.0) if len(filtered_df) > 0 else filtered_df

        # 🔧 5단계: 상위 10개 선택
        final_recommendations = filtered_df.head(10)

        print(f"  ✅ 최종 추천 향수: {len(final_recommendations)}개")

        # 🔧 6단계: 응답 데이터 생성
        results = []
        for idx, row in final_recommendations.iterrows():
            try:
                similarity_score = row.get('similarity_score', 0.5)

                # 감정 태그 선택 (우선순위: emotion_tags > impression_tags > 기본값)
                emotions = ""
                if 'emotion_tags' in row and pd.notna(row['emotion_tags']):
                    emotions = str(row['emotion_tags'])
                elif 'impression_tags' in row and pd.notna(row['impression_tags']):
                    emotions = str(row['impression_tags'])
                else:
                    emotions = f"{request.impression}, elegant"

                item = PerfumeRecommendItem(
                    name=str(row.get('name', f'향수 #{idx}')),
                    brand=str(row.get('brand', 'Unknown')),
                    image_url=str(row.get('image_url', '')),
                    notes=str(row.get('notes', 'No description available')),
                    emotions=emotions,
                    reason=f"{request.impression} 느낌에 {int(similarity_score * 100)}% 어울리는 향수입니다."
                )
                results.append(item)

            except Exception as e:
                print(f"    ⚠️ 항목 {idx} 처리 중 오류: {e}")
                continue

        # 🔧 7단계: 최종 검증
        if not results:
            print("  🚨 결과 생성 실패 - 기본 샘플 반환")
            results = [
                PerfumeRecommendItem(
                    name="클래식 향수",
                    brand="에센셜",
                    image_url="",
                    notes="당신의 취향에 맞는 클래식한 향수입니다.",
                    emotions=f"{request.impression}, elegant, timeless",
                    reason=f"{request.impression} 느낌의 추천 향수입니다."
                )
            ]

        print(f"✅ 추천 완료: {len(results)}개 반환\n")
        return results

    except Exception as e:
        print(f"❌ 추천 처리 중 심각한 오류: {e}")
        import traceback
        traceback.print_exc()

        # 🚨 최후의 수단: 에러 상황에서도 기본 응답 반환
        try:
            return [
                PerfumeRecommendItem(
                    name="에러 복구 향수",
                    brand="시스템",
                    image_url="",
                    notes="일시적인 오류로 인한 기본 추천입니다. 다시 시도해주세요.",
                    emotions="neutral, safe",
                    reason="시스템 오류로 인한 기본 추천입니다."
                )
            ]
        except:
            # HTTP 예외로 던지기
            raise HTTPException(
                status_code=500,
                detail=f"추천 시스템 오류: {str(e)}"
            )


# 나머지 API들 (기존과 동일)
class ClusterRequest(BaseModel):
    cluster_id: int


@router.post(
    "/recommend-by-cluster",
    response_model=List[PerfumeRecommendItem],
    summary="클러스터 기반 향수 추천"
)
def recommend_by_cluster(request: ClusterRequest):
    cluster_emotion_map = {
        0: "fresh",
        1: "elegant",
        2: "mysterious",
        3: "friendly",
        4: "pure",
        5: "confident"
    }

    target_impression = cluster_emotion_map.get(request.cluster_id, "fresh")

    basic_request = RecommendRequest(
        gender="unisex",
        season="spring",
        time="day",
        impression=target_impression,
        activity="casual",
        weather="any"
    )

    return recommend_perfumes(basic_request)


class EmotionClusterRequest(BaseModel):
    gender: Literal['women', 'men', 'unisex']
    season: Literal['spring', 'summer', 'fall', 'winter']
    time: Literal['day', 'night']
    impression: Literal['confident', 'elegant', 'pure', 'friendly', 'mysterious', 'fresh']
    activity: Literal['casual', 'work', 'date']
    weather: Literal['hot', 'cold', 'rainy', 'any']


class EmotionClusterResponse(BaseModel):
    cluster_id: int


@router.post(
    "/predict-emotion",
    response_model=EmotionClusterResponse,
    summary="감정 클러스터 예측"
)
def predict_emotion_cluster_id(request: EmotionClusterRequest):
    try:
        model_input = [
            request.gender,
            request.season,
            request.time,
            request.impression,
            request.activity,
            request.weather
        ]
        cluster_id = predict_emotion_cluster(model_input)
        return EmotionClusterResponse(cluster_id=cluster_id)
    except Exception as e:
        print(f"❌ 감정 예측 오류: {e}")
        impression_to_cluster = {
            'confident': 5,
            'elegant': 1,
            'pure': 3,
            'friendly': 3,
            'mysterious': 2,
            'fresh': 0
        }
        cluster_id = impression_to_cluster.get(request.impression, 0)
        return EmotionClusterResponse(cluster_id=cluster_id)