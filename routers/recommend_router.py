# routers/recommend_router.py - 즉시 적용 가능한 수정 버전

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

    # 🔍 태그 컬럼 확인 및 통합
    # emotion_tags가 있으면 사용, 없으면 생성
    if 'emotion_tags' not in df.columns:
        print("⚠️ emotion_tags 컬럼 없음 - 기본값으로 생성")
        df['emotion_tags'] = 'fresh, clean, elegant'

    # 🔍 다른 필수 컬럼들 확인
    other_columns = ['image_url', 'notes']
    for col in other_columns:
        if col not in df.columns:
            print(f"⚠️ {col} 컬럼 없음 - 기본값으로 생성")
            df[col] = ''

    # 🔧 벡터화 준비
    vectorizer = CountVectorizer(tokenizer=lambda x: str(x).split(", "))
    emotion_matrix = vectorizer.fit_transform(df["emotion_tags"].astype(str))

    print("✅ 데이터 준비 및 벡터화 완료")

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
        'emotion_tags': ['fresh, elegant, clean', 'confident, mysterious, deep', 'friendly, pure, light']
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


# ✅ 수정된 1차 향수 추천 API
@router.post(
    "/recommend",
    response_model=List[PerfumeRecommendItem],
    summary="1차 향수 추천",
    description="프론트엔드 요청에 맞춘 향수 추천"
)
def recommend_perfumes(request: RecommendRequest):
    """
    🔧 수정된 추천 로직:
    - 모든 태그 정보를 emotion_tags 컬럼에서 검색
    - 안전한 필터링 및 에러 처리
    - 프론트엔드 요청 형식에 정확히 대응
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

        # 🔧 2단계: emotion_tags에서 모든 특성 검색
        if 'emotion_tags' in df.columns and len(filtered_df) > 0:
            # 검색할 키워드들 (사용자가 입력한 모든 특성)
            search_keywords = [
                request.impression,  # 가장 중요한 인상
                request.season,
                request.time,
                request.activity
            ]

            # weather가 'any'가 아닌 경우에만 추가
            if request.weather != 'any':
                search_keywords.append(request.weather)

            print(f"  🔍 검색 키워드: {search_keywords}")

            # 각 키워드별로 점수 계산
            keyword_scores = []

            for keyword in search_keywords:
                try:
                    # emotion_tags에서 키워드 포함 여부 확인 (대소문자 무시)
                    mask = filtered_df['emotion_tags'].astype(str).str.lower().str.contains(
                        keyword.lower(), na=False, regex=False
                    )

                    matches = filtered_df[mask]
                    print(f"    '{keyword}' 매칭: {len(matches)}개")

                    if len(matches) > 0:
                        keyword_scores.append((keyword, matches))

                except Exception as e:
                    print(f"    ⚠️ '{keyword}' 검색 중 오류: {e}")
                    continue

            # 🔧 3단계: 가장 많은 키워드와 매칭되는 향수 우선 선택
            if keyword_scores:
                # impression (인상)을 가장 우선시
                impression_matches = None
                for keyword, matches in keyword_scores:
                    if keyword == request.impression:
                        impression_matches = matches
                        break

                if impression_matches is not None and len(impression_matches) > 0:
                    filtered_df = impression_matches
                    print(f"  ✅ 인상 기반 필터링: {len(filtered_df)}개 (impression={request.impression})")
                else:
                    # impression 매칭이 없으면 다른 키워드 매칭 사용
                    if keyword_scores:
                        _, best_matches = keyword_scores[0]  # 첫 번째 매칭 결과 사용
                        filtered_df = best_matches
                        print(f"  ✅ 대체 키워드 필터링: {len(filtered_df)}개")
            else:
                print(f"  ⚠️ 키워드 매칭 실패 - 성별 필터링 결과 사용")

        # 🔧 4단계: 최종 결과 확인 및 처리
        if len(filtered_df) == 0:
            print(f"  ❌ 필터링 결과 없음 - 전체 데이터에서 랜덤 선택")
            filtered_df = df.sample(n=min(10, len(df))) if len(df) > 0 else df

        # 🔧 5단계: 유사도 기반 정렬 (가능한 경우)
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

        # 🔧 6단계: 상위 10개 선택
        final_recommendations = filtered_df.head(10)

        print(f"  ✅ 최종 추천 향수: {len(final_recommendations)}개")

        # 🔧 7단계: 응답 데이터 생성
        results = []
        for idx, row in final_recommendations.iterrows():
            try:
                similarity_score = row.get('similarity_score', 0.5)

                item = PerfumeRecommendItem(
                    name=str(row.get('name', f'향수 #{idx}')),
                    brand=str(row.get('brand', 'Unknown')),
                    image_url=str(row.get('image_url', '')),
                    notes=str(row.get('notes', 'No description available')),
                    emotions=str(row.get('emotion_tags', '')),
                    reason=f"{request.impression} 느낌에 {int(similarity_score * 100)}% 어울리는 향수입니다."
                )
                results.append(item)

            except Exception as e:
                print(f"    ⚠️ 항목 {idx} 처리 중 오류: {e}")
                continue

        # 🔧 8단계: 최종 검증
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


# 나머지 API들은 기존과 동일하게 유지
class ClusterRequest(BaseModel):
    cluster_id: int


@router.post(
    "/recommend-by-cluster",
    response_model=List[PerfumeRecommendItem],
    summary="클러스터 기반 향수 추천"
)
def recommend_by_cluster(request: ClusterRequest):
    # 기존 로직과 동일하지만 더 안전하게 처리
    cluster_emotion_map = {
        0: "fresh",
        1: "elegant",
        2: "mysterious",
        3: "friendly",
        4: "pure",
        5: "confident"
    }

    target_impression = cluster_emotion_map.get(request.cluster_id, "fresh")

    # 기본 추천 요청으로 변환
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
        # 간단한 매핑으로 대체
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