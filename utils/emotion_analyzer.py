# utils/emotion_analyzer.py - Whiff 감정 분석 시스템

import re
import logging
import json
import os
import asyncio
import time
from typing import List, Dict, Tuple, Optional, Union, Any
from datetime import datetime
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)


class EmotionAnalyzer:
    """
    Whiff 시향일기 텍스트의 감정 분석 및 태그 생성 서비스

    Features:
    - 룰 기반 감정 분석 (현재 버전)
    - AI 모델 v2 준비 구조
    - 8개 핵심 감정 지원
    - 향수 도메인 특화 키워드
    - 폴백 메커니즘
    - 성능 모니터링
    """

    def __init__(self):
        """감정 분석기 초기화"""
        logger.info("🎭 감정 분석기 초기화 시작...")

        # 🎯 감정별 태그 매핑 (초기 모델 버전 - 확장 가능)
        self.emotion_to_tags = {
            "기쁨": ["#joyful", "#bright", "#citrus", "#happy", "#cheerful"],
            "불안": ["#nervous", "#sharp", "#spicy", "#anxious", "#tense"],
            "당황": ["#confused", "#mild", "#powdery", "#surprised", "#bewildered"],
            "분노": ["#angry", "#hot", "#burntwood", "#intense", "#fiery"],
            "상처": ["#hurt", "#cool", "#woody", "#sad", "#melancholy"],
            "슬픔": ["#sad", "#deep", "#musk", "#blue", "#tearful"],
            "우울": ["#depressed", "#dark", "#leather", "#gloomy", "#heavy"],
            "흥분": ["#excited", "#fresh", "#green", "#energetic", "#vibrant"]
        }

        # 🔍 룰 기반 감정 키워드 사전 (향수 도메인 특화)
        self.emotion_keywords = {
            "기쁨": [
                # 직접적 감정 표현
                "좋아", "행복", "기뻐", "즐거워", "만족", "완벽", "최고", "사랑",
                # 향수 관련 긍정 표현
                "상쾌", "밝은", "화사", "싱그러운", "상큼", "달콤", "포근", "따뜻",
                "사랑스러운", "예쁜", "고급스러운", "우아한", "세련된", "매력적",
                # 감각적 표현
                "부드러운", "은은한", "깔끔한", "깨끗한", "청량한", "시원한"
            ],
            "불안": [
                # 직접적 감정 표현
                "불안", "걱정", "긴장", "떨려", "두려운", "무서운", "조마조마",
                # 향수 관련 부정 표현
                "어색", "부담", "압박", "스트레스", "불편", "어색해",
                "이상해", "어색한", "답답", "무거운"
            ],
            "당황": [
                # 직접적 감정 표현
                "당황", "놀란", "혼란", "어리둥절", "멍한", "모르겠다", "헷갈려",
                # 예상과 다른 경험
                "이상", "예상과 달라", "의외", "신기", "특이", "독특",
                "예상못한", "뜻밖의", "갑작스러운"
            ],
            "분노": [
                # 직접적 감정 표현
                "화가", "짜증", "열받", "분노", "격정", "싫어", "별로", "최악",
                # 향수 관련 강한 부정
                "자극적", "강렬", "과해", "부담스러워", "독해", "역겨운",
                "끔찍", "못참겠", "견딜수없", "극혐"
            ],
            "상처": [
                # 직접적 감정 표현
                "상처", "아픈", "서운", "실망", "아쉬워", "슬픈", "힘든",
                "섭섭", "마음아픈", "쓸쓸",
                # 그리움과 연관
                "그리운", "그립", "애틋", "안타까운", "아련한"
            ],
            "슬픔": [
                # 직접적 감정 표현
                "슬퍼", "눈물", "애절", "처량", "고독", "외로운", "쓸쓸",
                "먹먹", "찡한", "울컥",
                # 깊은 감정
                "진한", "깊은", "차가운", "무거운", "침울한", "암울한"
            ],
            "우울": [
                # 직접적 감정 표현
                "우울", "답답", "무기력", "절망", "어둠", "침울", "멜랑콜리",
                "블루", "그늘진", "어두운", "막막한",
                # 깊은 우울감
                "절망적", "희망없는", "의욕없는", "공허한"
            ],
            "흥분": [
                # 직접적 감정 표현
                "흥분", "신나", "두근", "설렘", "활기", "생동감", "에너지",
                "활발", "역동적", "펄떡",
                # 향수 관련 활기
                "톡톡", "팡팡", "생생한", "활력", "젊은", "발랄한"
            ]
        }

        # 🌸 향수 도메인 특화 컨텍스트 키워드
        self.perfume_context_keywords = {
            "positive_intensity": {
                "mild": ["은은", "부드러운", "가벼운", "살짝", "은근"],
                "medium": ["적당", "괜찮은", "무난한", "균형잡힌"],
                "strong": ["진한", "강한", "깊은", "풍부한", "농축된"]
            },
            "positive_quality": [
                "좋아", "마음에 들어", "향이 좋아", "예뻐", "고급스러워", "우아해",
                "세련된", "매력적", "포근해", "따뜻해", "시원해", "청량해", "상큼해",
                "달콤해", "부드러워", "은은해", "깔끔해", "깨끗해", "fresh", "nice"
            ],
            "negative_quality": [
                "싫어", "별로", "아쉬워", "이상해", "어색해", "부담스러워", "과해",
                "단조로워", "밋밋해", "독해", "자극적", "역겨운", "끔찍"
            ],
            "intensity_negative": [
                "진해", "약해", "세", "강해", "독해", "자극적", "압도적"
            ],
            "temporal_context": [
                "처음", "첫", "나중", "시간지나", "지속", "변화", "바뀌"
            ]
        }

        # 🎯 모델 상태 관리
        self.model_loaded = False
        self.model = None
        self.tokenizer = None
        self.model_version = "v2_dev"  # 개발 중 버전
        self.analysis_count = 0
        self.performance_stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "average_response_time": 0.0,
            "method_distribution": {"rule_based": 0, "ai_model": 0},
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0}
        }

        logger.info("✅ 감정 분석기 초기화 완료")
        logger.info(f"  - 지원 감정: {list(self.emotion_to_tags.keys())}")
        logger.info(f"  - 모델 버전: {self.model_version} (개발 중)")
        logger.info(f"  - 총 키워드: {sum(len(keywords) for keywords in self.emotion_keywords.values())}개")

    async def analyze_emotion(self, text: str, use_model: bool = True) -> Dict[str, Any]:
        """
        텍스트 감정 분석 (모델 우선, 실패 시 룰 기반 폴백)

        Args:
            text: 분석할 텍스트
            use_model: AI 모델 사용 여부 (False면 룰 기반만 사용)

        Returns:
            분석 결과 딕셔너리
        """
        start_time = time.time()
        self.analysis_count += 1

        logger.info(f"🎭 감정 분석 시작 (#{self.analysis_count})")
        logger.info(f"  - 텍스트 길이: {len(text)}자")
        logger.info(f"  - 모델 사용: {'✅' if use_model else '❌'}")

        # 입력 검증
        if not text or not text.strip():
            return self._create_empty_result()

        if len(text) > 2000:
            logger.warning(f"⚠️ 텍스트가 너무 깁니다: {len(text)}자")
            return self._create_error_result("text_too_long", "텍스트가 2000자를 초과합니다.")

        try:
            # 🤖 AI 모델 분석 시도 (개발 완료 후)
            if use_model and self._is_model_available():
                try:
                    logger.info(f"🤖 AI 모델 v{self.model_version} 분석 시작...")
                    model_result = await self._analyze_with_model(text)

                    if model_result.get("success"):
                        response_time = time.time() - start_time
                        self._update_performance_stats(model_result, response_time)

                        logger.info(f"✅ AI 모델 분석 완료 (소요시간: {response_time:.3f}초)")
                        logger.info(f"  - 감정: {model_result.get('primary_emotion')}")
                        logger.info(f"  - 신뢰도: {model_result.get('confidence', 0):.3f}")

                        return model_result
                    else:
                        logger.warning("⚠️ AI 모델 분석 실패, 룰 기반으로 폴백")

                except Exception as e:
                    logger.error(f"❌ AI 모델 분석 중 오류: {e}")

            # 📋 룰 기반 분석 (폴백 또는 기본)
            logger.info(f"📋 룰 기반 감정 분석 시작...")
            rule_result = await self._analyze_with_rules(text)

            response_time = time.time() - start_time
            self._update_performance_stats(rule_result, response_time)

            logger.info(f"✅ 룰 기반 분석 완료 (소요시간: {response_time:.3f}초)")
            logger.info(f"  - 감정: {rule_result.get('primary_emotion')}")
            logger.info(f"  - 신뢰도: {rule_result.get('confidence', 0):.3f}")

            return rule_result

        except Exception as e:
            logger.error(f"❌ 감정 분석 중 예외 발생: {e}")
            return self._create_error_result("analysis_exception", str(e))

    def _is_model_available(self) -> bool:
        """AI 모델 사용 가능 여부 확인"""
        # 🚧 현재는 모델 개발 중이므로 False 반환
        # 모델 완성 후 실제 로딩 로직으로 교체
        return self.model_loaded and self.model is not None

    async def _analyze_with_model(self, text: str) -> Dict[str, Any]:
        """AI 모델을 사용한 감정 분석 (미래 구현)"""
        # 🚧 모델 개발 완료 후 구현 예정
        try:
            # TODO: 실제 모델 추론 로직
            # 1. 텍스트 전처리
            # processed_text = self._preprocess_text(text)

            # 2. 토크나이징
            # tokenized = self.tokenizer.transform([processed_text])

            # 3. 모델 추론 실행
            # prediction = self.model.predict(tokenized)
            # probabilities = self.model.predict_proba(tokenized)[0]

            # 4. 결과 후처리
            # emotion_idx = np.argmax(prediction)
            # confidence = float(max(probabilities))

            # 임시로 개발 중 상태 반환
            await asyncio.sleep(0.05)  # 모델 추론 시뮬레이션

            return {
                "success": False,
                "reason": "model_under_development",
                "message": f"AI 모델 v{self.model_version}는 현재 개발 중입니다."
            }

        except Exception as e:
            logger.error(f"❌ 모델 추론 중 오류: {e}")
            return {"success": False, "reason": "model_error", "message": str(e)}

    async def _analyze_with_rules(self, text: str) -> Dict[str, Any]:
        """룰 기반 감정 분석 (향수 도메인 특화)"""
        try:
            text_lower = text.lower().strip()
            text_words = text.split()

            # 🔍 1단계: 기본 감정 키워드 매칭
            emotion_scores = {}
            keyword_matches = {}
            total_keywords_found = 0

            for emotion, keywords in self.emotion_keywords.items():
                score = 0
                matched_keywords = []

                for keyword in keywords:
                    # 정확한 단어 매칭 (부분 문자열이 아닌)
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    matches = len(re.findall(pattern, text_lower))

                    if matches > 0:
                        # 키워드 중요도에 따른 가중치 적용
                        weight = self._get_keyword_weight(keyword, emotion)
                        score += matches * weight
                        matched_keywords.extend([keyword] * matches)
                        total_keywords_found += matches

                if score > 0:
                    emotion_scores[emotion] = score
                    keyword_matches[emotion] = matched_keywords

            # 🌸 2단계: 향수 도메인 컨텍스트 보정
            context_boost = self._analyze_perfume_context(text_lower, text_words)

            # 컨텍스트 보정 적용
            for emotion, boost in context_boost.items():
                if emotion in emotion_scores:
                    emotion_scores[emotion] += boost
                elif boost > 0.5:  # 충분히 강한 컨텍스트 신호
                    emotion_scores[emotion] = boost
                    keyword_matches[emotion] = ["context_boost"]

            # 📊 3단계: 결과 계산 및 정규화
            if emotion_scores:
                # 텍스트 길이에 따른 정규화
                normalization_factor = max(len(text_words), 1)
                normalized_scores = {}

                for emotion, score in emotion_scores.items():
                    # 정규화 점수 계산 (0.0 ~ 1.0)
                    normalized_score = min(score / normalization_factor * 1.5, 1.0)
                    normalized_scores[emotion] = normalized_score

                # 최고 점수 감정 선택
                primary_emotion = max(normalized_scores.keys(),
                                      key=lambda x: normalized_scores[x])
                confidence = normalized_scores[primary_emotion]

                # 신뢰도 보정 (키워드 매칭이 많을수록 신뢰도 증가)
                confidence_boost = min(total_keywords_found * 0.1, 0.3)
                final_confidence = min(confidence + confidence_boost, 1.0)

                # 감정 태그 생성
                emotion_tags = self.emotion_to_tags.get(primary_emotion, ["#neutral"])

                return {
                    "success": True,
                    "method": "rule_based",
                    "primary_emotion": primary_emotion,
                    "confidence": round(final_confidence, 3),
                    "emotion_tags": emotion_tags,
                    "analysis_details": {
                        "raw_scores": emotion_scores,
                        "normalized_scores": normalized_scores,
                        "keyword_matches": keyword_matches,
                        "context_boost": context_boost,
                        "total_keywords": total_keywords_found,
                        "text_length": len(text),
                        "word_count": len(text_words)
                    },
                    "analyzed_at": datetime.now().isoformat(),
                    "analyzer_version": "rule_based_v1.2"
                }
            else:
                # 키워드 매칭이 없는 경우
                return self._create_neutral_result("no_emotion_keywords")

        except Exception as e:
            logger.error(f"❌ 룰 기반 분석 중 오류: {e}")
            return self._create_error_result("rule_analysis_error", str(e))

    def _get_keyword_weight(self, keyword: str, emotion: str) -> float:
        """키워드 중요도 가중치 계산"""
        # 감정별 핵심 키워드에 높은 가중치
        high_weight_keywords = {
            "기쁨": ["좋아", "행복", "사랑", "완벽", "최고"],
            "불안": ["불안", "걱정", "두려운", "부담"],
            "당황": ["당황", "놀란", "혼란", "의외"],
            "분노": ["화가", "짜증", "싫어", "최악"],
            "상처": ["상처", "아픈", "실망", "그리운"],
            "슬픔": ["슬퍼", "눈물", "외로운", "쓸쓸"],
            "우울": ["우울", "절망", "무기력", "어둠"],
            "흥분": ["흥분", "신나", "설렘", "에너지"]
        }

        if keyword in high_weight_keywords.get(emotion, []):
            return 1.5  # 핵심 키워드 가중치
        elif len(keyword) >= 3:
            return 1.2  # 긴 키워드 가중치
        else:
            return 1.0  # 기본 가중치

    def _analyze_perfume_context(self, text_lower: str, text_words: List[str]) -> Dict[str, float]:
        """향수 도메인 특화 컨텍스트 분석"""
        context_boost = {}

        # 긍정적 품질 표현 감지
        positive_quality_count = 0
        for keyword in self.perfume_context_keywords["positive_quality"]:
            positive_quality_count += text_lower.count(keyword)

        # 부정적 품질 표현 감지
        negative_quality_count = 0
        for keyword in self.perfume_context_keywords["negative_quality"]:
            negative_quality_count += text_lower.count(keyword)

        # 강도 관련 부정 표현
        intensity_negative_count = 0
        for keyword in self.perfume_context_keywords["intensity_negative"]:
            intensity_negative_count += text_lower.count(keyword)

        # 🌸 컨텍스트 기반 감정 보정
        if positive_quality_count > 0:
            boost_strength = min(positive_quality_count * 0.8, 2.0)
            context_boost["기쁨"] = boost_strength

            # 긍정적 표현이 매우 강한 경우 흥분도 추가
            if positive_quality_count >= 2:
                context_boost["흥분"] = boost_strength * 0.6

        if negative_quality_count > 0:
            boost_strength = min(negative_quality_count * 0.7, 1.8)
            # 부정적 표현의 강도에 따라 다른 감정 배정
            if negative_quality_count >= 2:
                context_boost["분노"] = boost_strength
            else:
                context_boost["상처"] = boost_strength * 0.8

        if intensity_negative_count > 0:
            boost_strength = min(intensity_negative_count * 0.6, 1.5)
            context_boost["불안"] = boost_strength

        # 시간적 맥락 분석 (변화 표현)
        temporal_keywords = ["처음", "첫", "나중", "시간지나", "변화"]
        temporal_count = sum(text_lower.count(kw) for kw in temporal_keywords)
        if temporal_count > 0:
            context_boost["당황"] = min(temporal_count * 0.5, 1.0)

        return context_boost

    def _create_empty_result(self) -> Dict[str, Any]:
        """빈 텍스트에 대한 결과"""
        return {
            "success": True,
            "method": "validation",
            "primary_emotion": "중립",
            "confidence": 0.0,
            "emotion_tags": ["#neutral"],
            "reason": "empty_text",
            "message": "분석할 텍스트가 비어있습니다.",
            "analyzed_at": datetime.now().isoformat()
        }

    def _create_neutral_result(self, reason: str) -> Dict[str, Any]:
        """중립 감정 결과"""
        return {
            "success": True,
            "method": "rule_based",
            "primary_emotion": "중립",
            "confidence": 0.3,
            "emotion_tags": ["#neutral", "#calm"],
            "reason": reason,
            "message": "명확한 감정 신호가 감지되지 않았습니다.",
            "analyzed_at": datetime.now().isoformat()
        }

    def _create_error_result(self, error_type: str, message: str) -> Dict[str, Any]:
        """에러 결과"""
        return {
            "success": False,
            "error_type": error_type,
            "message": message,
            "primary_emotion": "오류",
            "confidence": 0.0,
            "emotion_tags": ["#error"],
            "analyzed_at": datetime.now().isoformat()
        }

    def _update_performance_stats(self, result: Dict[str, Any], response_time: float):
        """성능 통계 업데이트"""
        self.performance_stats["total_analyses"] += 1

        if result.get("success"):
            self.performance_stats["successful_analyses"] += 1

            # 방법별 분포 업데이트
            method = result.get("method", "unknown")
            if method in self.performance_stats["method_distribution"]:
                self.performance_stats["method_distribution"][method] += 1

            # 신뢰도 분포 업데이트
            confidence = result.get("confidence", 0.0)
            if confidence >= 0.7:
                self.performance_stats["confidence_distribution"]["high"] += 1
            elif confidence >= 0.4:
                self.performance_stats["confidence_distribution"]["medium"] += 1
            else:
                self.performance_stats["confidence_distribution"]["low"] += 1

        # 평균 응답 시간 업데이트
        total_time = (self.performance_stats["average_response_time"] *
                      (self.performance_stats["total_analyses"] - 1) + response_time)
        self.performance_stats["average_response_time"] = total_time / self.performance_stats["total_analyses"]

    def get_supported_emotions(self) -> List[str]:
        """지원하는 감정 목록 반환"""
        return list(self.emotion_to_tags.keys())

    def get_emotion_tags(self, emotion: str) -> List[str]:
        """특정 감정의 태그 목록 반환"""
        return self.emotion_to_tags.get(emotion, ["#neutral"])

    def get_analysis_stats(self) -> Dict[str, Any]:
        """분석 시스템 상태 정보"""
        success_rate = 0.0
        if self.performance_stats["total_analyses"] > 0:
            success_rate = (self.performance_stats["successful_analyses"] /
                            self.performance_stats["total_analyses"] * 100)

        return {
            # 기본 정보
            "model_loaded": self.model_loaded,
            "model_version": self.model_version,
            "supported_emotions": len(self.emotion_to_tags),
            "total_keywords": sum(len(keywords) for keywords in self.emotion_keywords.values()),
            "analysis_methods": ["rule_based"] + (["ai_model"] if self.model_loaded else []),

            # 성능 통계
            "performance": {
                "total_analyses": self.performance_stats["total_analyses"],
                "successful_analyses": self.performance_stats["successful_analyses"],
                "success_rate": round(success_rate, 2),
                "average_response_time": round(self.performance_stats["average_response_time"], 3),
                "method_distribution": self.performance_stats["method_distribution"],
                "confidence_distribution": self.performance_stats["confidence_distribution"]
            },

            # 감정 목록
            "emotion_list": list(self.emotion_to_tags.keys()),
            "emotion_tags_count": {emotion: len(tags) for emotion, tags in self.emotion_to_tags.items()},

            # 시스템 정보
            "system_info": {
                "max_text_length": 2000,
                "supported_languages": ["한국어"],
                "domain_specialization": "향수_리뷰",
                "last_updated": datetime.now().isoformat()
            }
        }


# 🌟 전역 감정 분석기 인스턴스
emotion_analyzer = EmotionAnalyzer()


# 🧪 테스트 및 디버깅 함수들
async def test_emotion_analyzer():
    """감정 분석기 테스트 함수"""
    print("🧪 감정 분석기 테스트 시작...\n")

    test_cases = [
        "이 향수 정말 좋아요! 달콤하고 상큼해서 기분이 좋아져요.",
        "향이 너무 진해서 별로예요. 좀 부담스럽네요.",
        "처음 맡았을 때 놀랐어요. 예상과 완전 달라서 당황스러웠어요.",
        "이 향수를 맡으면 옛날 생각이 나서 슬퍼져요.",
        "향수가 너무 자극적이어서 화가 나요. 최악이에요.",
        "새로운 향수를 발견해서 너무 신나요! 에너지가 넘쳐요.",
        "향이 은은하고 깔끔해서 마음에 들어요.",
        ""  # 빈 텍스트 테스트
    ]

    for i, text in enumerate(test_cases, 1):
        print(f"--- 테스트 {i} ---")
        print(f"입력: {text if text else '(빈 텍스트)'}")

        result = await emotion_analyzer.analyze_emotion(text)

        print(f"결과: {result['primary_emotion']} (신뢰도: {result['confidence']:.3f})")
        print(f"태그: {result['emotion_tags']}")
        print(f"방법: {result['method']}")
        print()

    # 성능 통계 출력
    stats = emotion_analyzer.get_analysis_stats()
    print("📊 성능 통계:")
    print(f"  총 분석: {stats['performance']['total_analyses']}회")
    print(f"  성공률: {stats['performance']['success_rate']}%")
    print(f"  평균 응답시간: {stats['performance']['average_response_time']:.3f}초")
    print()

    print("✅ 테스트 완료!")


if __name__ == "__main__":
    # 직접 실행 시 테스트
    asyncio.run(test_emotion_analyzer())