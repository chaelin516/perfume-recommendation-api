# routers/diary_router.py - 룰 기반 감정분석 시스템 (토큰 인증 제거)

from fastapi import APIRouter, Query, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from schemas.diary import DiaryCreateRequest, DiaryResponse
from schemas.common import BaseResponse
from datetime import datetime, date
from typing import Optional, List, Dict, Any
import os
import json
import uuid
import logging
import asyncio
import time
from collections import Counter
from pydantic import BaseModel, Field
import re

# 🎭 로거 설정
logger = logging.getLogger("diary_router")

# ✅ 라우터 생성
router = APIRouter(prefix="/diaries", tags=["Diary"])

# 📂 데이터 파일 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIARY_PATH = os.path.join(BASE_DIR, "../data/diary_data.json")

# 🎯 룰 기반 감정 분석 사전 정의
EMOTION_RULES = {
    "기쁨": {
        "keywords": [
            "좋", "훌륭", "향긋", "달콤", "상큼", "깔끔", "사랑", "완벽", "최고", "멋진",
            "환상적", "놀라운", "아름다운", "우아한", "세련된", "고급스러운", "매력적",
            "기분좋", "행복", "즐거운", "만족", "감동", "황홀", "반함", "좋아해", "마음에 들",
            "포근", "따뜻", "편안", "부드러운", "은은한", "우아", "고혹적", "신비로운"
        ],
        "tags": ["#happy", "#joyful", "#pleasant", "#nice", "#satisfied", "#lovely"],
        "base_confidence": 0.7
    },
    "설렘": {
        "keywords": [
            "설레", "두근", "떨림", "기대", "호기심", "궁금", "신기", "새로운", "특별한",
            "독특한", "매혹적", "흥미로운", "재미있", "신선한", "생동감", "활기찬",
            "짜릿", "스릴", "흥분", "가슴이 뛰", "심장이", "떨려"
        ],
        "tags": ["#excited", "#curious", "#anticipation", "#thrilling", "#fascinating"],
        "base_confidence": 0.65
    },
    "평온": {
        "keywords": [
            "평온", "고요", "잔잔", "차분", "안정", "평화", "조용", "편안", "여유로운",
            "느긋", "릴렉스", "휴식", "치유", "힐링", "진정", "마음이 편해", "스트레스",
            "피로가 풀", "숨을 쉬기", "깊게 호흡", "명상", "사색", "생각에 잠기"
        ],
        "tags": ["#calm", "#peaceful", "#relaxed", "#healing", "#serene"],
        "base_confidence": 0.6
    },
    "자신감": {
        "keywords": [
            "자신감", "당당", "세련", "고급", "품격", "우아", "카리스마", "멋있", "섹시",
            "매력적", "강렬", "파워풀", "임팩트", "프로페셔널", "성숙한", "어른스러운"
        ],
        "tags": ["#confident", "#elegant", "#sophisticated", "#charismatic", "#powerful"],
        "base_confidence": 0.65
    },
    "활력": {
        "keywords": [
            "활력", "에너지", "생동감", "활기", "싱그러운", "상쾌", "시원", "청량감",
            "톡톡", "팝", "활발", "역동적", "젊은", "발랄", "명랑", "생기발랄"
        ],
        "tags": ["#energetic", "#fresh", "#vibrant", "#lively", "#dynamic"],
        "base_confidence": 0.6
    },
    "로맨틱": {
        "keywords": [
            "로맨틱", "낭만", "사랑", "달콤", "부드러운", "따뜻한", "포근", "감미로운",
            "달콤쌉쌀", "심쿵", "로맨스", "데이트", "연인", "커플", "달달한"
        ],
        "tags": ["#romantic", "#sweet", "#lovely", "#tender", "#affectionate"],
        "base_confidence": 0.7
    },
    "그리움": {
        "keywords": [
            "그리움", "향수", "추억", "그립", "옛날", "어릴적", "추상적", "몽환적",
            "아련", "쓸쓸", "서정적", "감성적", "애틋", "생각나", "기억"
        ],
        "tags": ["#nostalgic", "#memory", "#longing", "#sentimental", "#wistful"],
        "base_confidence": 0.6
    }
}

# 🌍 상황별 감정 부스터
CONTEXT_BOOSTERS = {
    "계절": {
        "봄": {"기쁨": 0.2, "활력": 0.15, "로맨틱": 0.1},
        "여름": {"활력": 0.25, "자신감": 0.15, "기쁨": 0.1},
        "가을": {"그리움": 0.2, "평온": 0.15, "로맨틱": 0.1},
        "겨울": {"평온": 0.2, "그리움": 0.15, "로맨틱": 0.1}
    },
    "시간": {
        "아침": {"활력": 0.2, "자신감": 0.15},
        "낮": {"기쁨": 0.15, "활력": 0.1},
        "저녁": {"로맨틱": 0.2, "평온": 0.15},
        "밤": {"그리움": 0.2, "평온": 0.15, "로맨틱": 0.1}
    },
    "상황": {
        "데이트": {"로맨틱": 0.3, "설렘": 0.2},
        "업무": {"자신감": 0.2, "활력": 0.15},
        "휴식": {"평온": 0.25, "기쁨": 0.1},
        "외출": {"활력": 0.15, "자신감": 0.1}
    }
}

# 🎨 향수 타입별 감정 매핑
PERFUME_TYPE_EMOTIONS = {
    "플로럴": ["로맨틱", "기쁨", "평온"],
    "시트러스": ["활력", "기쁨", "자신감"],
    "우디": ["자신감", "평온", "그리움"],
    "바닐라": ["평온", "로맨틱", "그리움"],
    "머스크": ["자신감", "로맨틱", "평온"],
    "프루티": ["기쁨", "활력", "설렘"]
}


def load_diary_data():
    """시향 일기 데이터 로딩"""
    if os.path.exists(DIARY_PATH):
        try:
            with open(DIARY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"✅ 시향 일기 데이터 로딩: {len(data)}개")
            return data
        except Exception as e:
            logger.error(f"❌ 시향 일기 데이터 로딩 실패: {e}")
    return []


def save_diary_data(data):
    """시향 일기 데이터 저장"""
    try:
        os.makedirs(os.path.dirname(DIARY_PATH), exist_ok=True)
        with open(DIARY_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"❌ 시향 일기 데이터 저장 실패: {e}")
        return False


def extract_context_from_text(text: str) -> Dict[str, str]:
    """텍스트에서 계절, 시간, 상황 정보 추출"""
    context = {"계절": None, "시간": None, "상황": None}
    text_lower = text.lower()

    # 계절 키워드
    season_keywords = {
        "봄": ["봄", "벚꽃", "개화", "따뜻해지", "신록"],
        "여름": ["여름", "덥", "시원", "해변", "바캉스", "휴가"],
        "가을": ["가을", "단풍", "쌀쌀", "선선", "추석"],
        "겨울": ["겨울", "춥", "눈", "크리스마스", "연말"]
    }

    # 시간 키워드
    time_keywords = {
        "아침": ["아침", "새벽", "출근", "모닝"],
        "낮": ["낮", "점심", "오후", "데이타임"],
        "저녁": ["저녁", "퇴근", "이브닝"],
        "밤": ["밤", "야간", "늦은", "자기전"]
    }

    # 상황 키워드
    situation_keywords = {
        "데이트": ["데이트", "만남", "연인", "커플"],
        "업무": ["회사", "업무", "미팅", "출근", "직장"],
        "휴식": ["휴식", "쉬", "여유", "릴렉스"],
        "외출": ["외출", "나가", "쇼핑", "친구"]
    }

    # 키워드 매칭
    for season, keywords in season_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            context["계절"] = season
            break

    for time_period, keywords in time_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            context["시간"] = time_period
            break

    for situation, keywords in situation_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            context["상황"] = situation
            break

    return context


def detect_perfume_type(perfume_name: str) -> str:
    """향수 이름으로부터 타입 추론"""
    perfume_lower = perfume_name.lower()

    type_keywords = {
        "플로럴": ["rose", "jasmine", "lily", "peony", "장미", "자스민", "백합"],
        "시트러스": ["lemon", "orange", "bergamot", "citrus", "레몬", "오렌지", "베르가못"],
        "우디": ["wood", "cedar", "sandalwood", "oak", "우드", "시더", "샌달우드"],
        "바닐라": ["vanilla", "바닐라"],
        "머스크": ["musk", "머스크"],
        "프루티": ["apple", "peach", "berry", "사과", "복숭아", "베리"]
    }

    for perfume_type, keywords in type_keywords.items():
        if any(keyword in perfume_lower for keyword in keywords):
            return perfume_type

    return "기타"


async def rule_based_emotion_analysis(content: str, perfume_name: str = "") -> Dict[str, Any]:
    """룰 기반 감정 분석 엔진"""
    try:
        logger.info(f"🎯 룰 기반 감정 분석 시작: 텍스트 길이 {len(content)}자")

        if not content or not content.strip():
            return {
                "success": False,
                "primary_emotion": "중립",
                "confidence": 0.3,
                "emotion_tags": ["#neutral"],
                "analysis_method": "no_content"
            }

        content_lower = content.lower()
        emotion_scores = {}

        # 각 감정별 점수 계산
        for emotion, rule in EMOTION_RULES.items():
            matched_keywords = []
            score = 0.0

            # 키워드 매칭
            for keyword in rule["keywords"]:
                if keyword in content_lower:
                    matched_keywords.append(keyword)
                    score += 1.0

            # 기본 신뢰도 적용
            if matched_keywords:
                confidence = min(rule["base_confidence"] + (len(matched_keywords) * 0.1), 0.95)
                emotion_scores[emotion] = {
                    "confidence": confidence,
                    "matched_keywords": matched_keywords,
                    "keyword_count": len(matched_keywords)
                }

        # 상황별 부스터 적용
        context = extract_context_from_text(content)
        for context_type, context_value in context.items():
            if context_value and context_type in CONTEXT_BOOSTERS:
                boosters = CONTEXT_BOOSTERS[context_type].get(context_value, {})
                for emotion, boost in boosters.items():
                    if emotion in emotion_scores:
                        emotion_scores[emotion]["confidence"] += boost
                        emotion_scores[emotion]["confidence"] = min(emotion_scores[emotion]["confidence"], 0.95)

        # 향수 타입별 감정 부스터
        perfume_type = detect_perfume_type(perfume_name)
        if perfume_type in PERFUME_TYPE_EMOTIONS:
            for emotion in PERFUME_TYPE_EMOTIONS[perfume_type]:
                if emotion in emotion_scores:
                    emotion_scores[emotion]["confidence"] += 0.1
                    emotion_scores[emotion]["confidence"] = min(emotion_scores[emotion]["confidence"], 0.95)

        # 주요 감정 결정
        if emotion_scores:
            primary_emotion = max(emotion_scores.keys(), key=lambda e: emotion_scores[e]["confidence"])
            confidence = emotion_scores[primary_emotion]["confidence"]

            # 감정 태그 생성 (상위 3개 감정)
            top_emotions = sorted(emotion_scores.items(), key=lambda x: x[1]["confidence"], reverse=True)[:3]
            emotion_tags = []
            for emotion, data in top_emotions:
                emotion_tags.extend(EMOTION_RULES[emotion]["tags"][:2])
        else:
            primary_emotion = "중립"
            confidence = 0.3
            emotion_tags = ["#neutral"]

        # 분석 결과 정리
        analysis_result = {
            "success": True,
            "primary_emotion": primary_emotion,
            "confidence": round(min(confidence, 0.95), 3),
            "emotion_tags": emotion_tags,
            "analysis_method": "rule_based",
            "emotion_scores": {k: round(v["confidence"], 3) for k, v in emotion_scores.items()},
            "context_detected": context,
            "perfume_type": perfume_type,
            "matched_keywords_summary": {
                emotion: data["matched_keywords"]
                for emotion, data in emotion_scores.items()
            }
        }

        logger.info(f"🎯 룰 기반 감정 분석 완료: {primary_emotion} ({confidence:.3f})")
        return analysis_result

    except Exception as e:
        logger.error(f"❌ 룰 기반 감정 분석 오류: {e}")
        return {
            "success": False,
            "primary_emotion": "중립",
            "confidence": 0.3,
            "emotion_tags": ["#neutral"],
            "analysis_method": "error_fallback",
            "error": str(e)
        }


# 전역 데이터
diary_data = load_diary_data()


# ✅ API 엔드포인트들

def get_default_user():
    """기본 사용자 정보 (토큰 없이 사용할 때)"""
    return {
        "uid": "anonymous_user",
        "name": "익명 사용자",
        "email": "",
        "picture": ""
    }


@router.post("/", summary="시향 일기 작성")
async def write_diary(
        entry: DiaryCreateRequest,
        background_tasks: BackgroundTasks
):
    """
    시향 일기 작성 (토큰 인증 없음)

    - user_id는 요청 데이터에서 받습니다
    - 토큰 인증이 제거되어 누구나 사용 가능합니다
    """
    try:
        # 기본 사용자 정보 설정 (토큰 없이 사용)
        user = get_default_user()

        # user_id가 요청에 있으면 사용, 없으면 기본값 사용
        user_id = getattr(entry, 'user_id', 'anonymous_user')
        if hasattr(entry, 'user_id') and entry.user_id:
            user_id = entry.user_id

        now = datetime.now().isoformat()
        diary_id = str(uuid.uuid4())

        logger.info(f"📝 새 일기 작성: {user_id} - {entry.perfume_name}")

        # 룰 기반 감정 분석
        initial_analysis = None
        if entry.content and entry.content.strip():
            try:
                initial_analysis = await asyncio.wait_for(
                    rule_based_emotion_analysis(entry.content, entry.perfume_name),
                    timeout=5.0  # 룰 기반이므로 더 빠름
                )
            except Exception as e:
                logger.error(f"❌ 감정 분석 오류: {e}")
                initial_analysis = {
                    "success": False,
                    "primary_emotion": "중립",
                    "confidence": 0.3,
                    "emotion_tags": ["#neutral"],
                    "analysis_method": "error"
                }

        # 일기 데이터 생성
        diary = {
            "id": diary_id,
            "user_id": user_id,
            "user_name": user.get("name", "익명 사용자"),
            "user_profile_image": user.get("picture", ""),
            "perfume_id": f"perfume_{entry.perfume_name.lower().replace(' ', '_')}",
            "perfume_name": entry.perfume_name,
            "brand": "Unknown Brand",
            "content": entry.content or "",
            "tags": entry.emotion_tags or [],
            "likes": 0,
            "comments": 0,
            "is_public": entry.is_public,
            "created_at": now,
            "updated_at": now,

            # 감정 분석 정보
            "emotion_analysis": initial_analysis,
            "primary_emotion": initial_analysis.get("primary_emotion", "중립") if initial_analysis else "중립",
            "emotion_confidence": initial_analysis.get("confidence", 0.0) if initial_analysis else 0.0,
            "emotion_tags_auto": initial_analysis.get("emotion_tags", []) if initial_analysis else [],
            "emotion_analysis_status": "completed" if initial_analysis and initial_analysis.get(
                "success") else "failed",
            "analysis_method": "rule_based"
        }

        # 태그 병합 (사용자 입력 태그 + 자동 분석 태그)
        if initial_analysis and initial_analysis.get("emotion_tags"):
            auto_tags = initial_analysis.get("emotion_tags", [])
            manual_tags = entry.emotion_tags or []
            diary["tags"] = list(set(manual_tags + auto_tags))

        # 저장
        diary_data.append(diary)
        save_diary_data(diary_data)

        return JSONResponse(
            status_code=200,
            content={
                "message": "시향 일기가 성공적으로 저장되었습니다.",
                "diary_id": diary_id,
                "user_id": user_id,
                "emotion_analysis": {
                    "status": diary["emotion_analysis_status"],
                    "method": "rule_based",
                    "primary_emotion": diary["primary_emotion"],
                    "confidence": diary["emotion_confidence"],
                    "context_detected": initial_analysis.get("context_detected", {}) if initial_analysis else {},
                    "perfume_type": initial_analysis.get("perfume_type", "기타") if initial_analysis else "기타"
                }
            }
        )

    except Exception as e:
        logger.error(f"❌ 일기 저장 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": f"일기 저장 중 오류: {str(e)}"}
        )


@router.get("/", summary="시향 일기 목록 조회")
async def get_diary_list(
        public: Optional[bool] = Query(None, description="공개 여부 필터"),
        page: Optional[int] = Query(1, description="페이지 번호"),
        size: Optional[int] = Query(10, description="페이지 크기"),
        keyword: Optional[str] = Query(None, description="검색 키워드")
):
    try:
        filtered_data = diary_data.copy()

        # 필터링
        if public is not None:
            filtered_data = [d for d in filtered_data if d.get("is_public") == public]

        if keyword:
            filtered_data = [d for d in filtered_data
                             if keyword.lower() in d.get("content", "").lower()
                             or keyword.lower() in d.get("perfume_name", "").lower()]

        # 정렬 및 페이징
        filtered_data.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        start = (page - 1) * size
        end = start + size
        paginated_data = filtered_data[start:end]

        # 응답 데이터 변환
        response_data = []
        for item in paginated_data:
            response_data.append({
                "id": item.get("id", ""),
                "user_name": item.get("user_name", "익명"),
                "perfume_name": item.get("perfume_name", ""),
                "content": item.get("content", ""),
                "tags": item.get("tags", []),
                "primary_emotion": item.get("primary_emotion", "중립"),
                "emotion_confidence": item.get("emotion_confidence", 0.0),
                "analysis_method": item.get("analysis_method", "rule_based"),
                "likes": item.get("likes", 0),
                "created_at": item.get("created_at", "")
            })

        return BaseResponse(
            message=f"시향 일기 목록 조회 성공 (총 {len(filtered_data)}개)",
            result={
                "diaries": response_data,
                "total_count": len(filtered_data),
                "page": page,
                "size": size,
                "has_next": end < len(filtered_data),
                "analysis_method": "rule_based"
            }
        )

    except Exception as e:
        logger.error(f"❌ 일기 목록 조회 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": f"서버 오류: {str(e)}"}
        )


@router.get("/{diary_id}", summary="특정 시향 일기 조회")
async def get_diary_detail(diary_id: str):
    """특정 시향 일기의 상세 정보를 조회합니다."""
    try:
        diary = next((d for d in diary_data if d.get("id") == diary_id), None)

        if not diary:
            raise HTTPException(status_code=404, detail="시향 일기를 찾을 수 없습니다.")

        return BaseResponse(
            message="시향 일기 조회 성공",
            result=diary
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 일기 상세 조회 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": f"서버 오류: {str(e)}"}
        )


@router.get("/stats/emotions", summary="감정 통계")
async def get_emotion_stats():
    """시향 일기의 감정 분석 통계를 반환합니다."""
    try:
        emotion_counts = Counter()
        total_diaries = len(diary_data)

        for diary in diary_data:
            primary_emotion = diary.get("primary_emotion", "중립")
            emotion_counts[primary_emotion] += 1

        emotion_stats = {}
        for emotion, count in emotion_counts.items():
            emotion_stats[emotion] = {
                "count": count,
                "percentage": round((count / total_diaries) * 100, 2) if total_diaries > 0 else 0
            }

        return BaseResponse(
            message="감정 통계 조회 성공",
            result={
                "total_diaries": total_diaries,
                "emotion_distribution": emotion_stats,
                "most_common_emotions": emotion_counts.most_common(5)
            }
        )

    except Exception as e:
        logger.error(f"❌ 감정 통계 조회 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"message": f"서버 오류: {str(e)}"}
        )