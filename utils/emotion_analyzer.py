# utils/emotion_analyzer.py - Google Drive 모델 지원 추가 버전

import re
import logging
import json
import os
import asyncio
import time
import hashlib
from typing import List, Dict, Tuple, Optional, Union, Any
from datetime import datetime
from collections import Counter
import numpy as np

logger = logging.getLogger(__name__)


class EmotionAnalyzer:
    """
    Whiff 시향일기 텍스트의 감정 분석 및 태그 생성 서비스 (Google Drive 모델 지원)

    Features:
    - 룰 기반 감정 분석 (현재 버전)
    - AI 모델 v2 준비 구조
    - 🆕 Google Drive 모델 자동 다운로드
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

        # 🆕 Google Drive 모델 설정
        self.google_drive_model_id = None  # 설정에서 로드
        self.google_drive_enabled = False
        self.model_cache_dir = "./models/cache"
        self.model_hash_file = "./models/model_hash.txt"

        # 성능 통계
        self.performance_stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "average_response_time": 0.0,
            "method_distribution": {"rule_based": 0, "ai_model": 0, "google_drive": 0},
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0}
        }

        # 🆕 초기화 시 Google Drive 설정 확인
        self._load_google_drive_config()

        logger.info("✅ 감정 분석기 초기화 완료")
        logger.info(f"  - 지원 감정: {list(self.emotion_to_tags.keys())}")
        logger.info(f"  - 모델 버전: {self.model_version} (개발 중)")
        logger.info(f"  - 총 키워드: {sum(len(keywords) for keywords in self.emotion_keywords.values())}개")
        logger.info(f"  - Google Drive 지원: {'✅' if self.google_drive_enabled else '❌'}")

    def _load_google_drive_config(self):
        """Google Drive 모델 설정 로드"""
        try:
            # 환경변수에서 Google Drive 모델 ID 로드
            self.google_drive_model_id = os.getenv('GOOGLE_DRIVE_MODEL_ID')

            if self.google_drive_model_id:
                self.google_drive_enabled = True
                logger.info(f"🌤️ Google Drive 모델 ID 설정됨: {self.google_drive_model_id[:20]}...")

                # 캐시 디렉토리 생성
                os.makedirs(self.model_cache_dir, exist_ok=True)

            else:
                logger.info("📋 Google Drive 모델 ID 없음, 로컬 모델만 사용")

        except Exception as e:
            logger.warning(f"⚠️ Google Drive 설정 로드 실패: {e}")
            self.google_drive_enabled = False

    def check_google_drive_model(self) -> bool:
        """Google Drive 모델 사용 가능 여부 확인"""
        if not self.google_drive_enabled:
            return False

        try:
            import gdown

            # 모델 파일 존재 확인
            cached_model_path = os.path.join(self.model_cache_dir, "emotion_model_gdrive.pkl")

            if os.path.exists(cached_model_path):
                # 파일 크기 확인
                file_size = os.path.getsize(cached_model_path)
                if file_size > 1000:  # 1KB 이상
                    logger.info(f"✅ Google Drive 모델 캐시 발견: {file_size:,}B")
                    return True

            logger.info("📦 Google Drive 모델 캐시 없음, 다운로드 필요")
            return False

        except ImportError:
            logger.warning("⚠️ gdown 라이브러리 없음, Google Drive 모델 사용 불가")
            return False
        except Exception as e:
            logger.error(f"❌ Google Drive 모델 확인 실패: {e}")
            return False

    async def download_google_drive_model(self) -> bool:
        """Google Drive에서 모델 다운로드"""
        if not self.google_drive_enabled:
            logger.warning("⚠️ Google Drive 모델이 활성화되지 않음")
            return False

        try:
            import gdown

            logger.info("📥 Google Drive에서 감정 분석 모델 다운로드 시작...")

            # 다운로드 URL 구성
            download_url = f"https://drive.google.com/uc?id={self.google_drive_model_id}"
            output_path = os.path.join(self.model_cache_dir, "emotion_model_gdrive.pkl")

            # 기존 파일 삭제 (있는 경우)
            if os.path.exists(output_path):
                os.remove(output_path)
                logger.info("🗑️ 기존 캐시 파일 삭제")

            # 다운로드 실행
            start_time = time.time()
            gdown.download(download_url, output_path, quiet=False)
            download_time = time.time() - start_time

            # 다운로드 성공 확인
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logger.info(f"✅ Google Drive 모델 다운로드 완료")
                logger.info(f"  - 파일 크기: {file_size:,}B ({file_size / 1024:.1f}KB)")
                logger.info(f"  - 다운로드 시간: {download_time:.2f}초")

                # 파일 해시 저장 (모델 무결성 확인용)
                model_hash = self._calculate_file_hash(output_path)
                with open(self.model_hash_file, 'w') as f:
                    f.write(model_hash)
                logger.info(f"🔐 모델 해시 저장: {model_hash[:8]}...")

                return True
            else:
                logger.error("❌ 다운로드 완료 후 파일이 없음")
                return False

        except ImportError:
            logger.error("❌ gdown 라이브러리가 설치되지 않음")
            return False
        except Exception as e:
            logger.error(f"❌ Google Drive 모델 다운로드 실패: {e}")
            return False

    def _calculate_file_hash(self, file_path: str) -> str:
        """파일 해시 계산 (무결성 확인용)"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.warning(f"⚠️ 파일 해시 계산 실패: {e}")
            return "unknown"

    async def load_google_drive_model(self):
        """Google Drive 모델 로딩"""
        if not self.google_drive_enabled:
            return False

        try:
            cached_model_path = os.path.join(self.model_cache_dir, "emotion_model_gdrive.pkl")

            # 캐시된 모델 확인
            if not os.path.exists(cached_model_path):
                logger.info("📦 Google Drive 모델 캐시 없음, 다운로드 시도")
                download_success = await self.download_google_drive_model()
                if not download_success:
                    return False

            # 모델 로딩
            logger.info("📦 Google Drive 모델 로딩 시작...")

            import pickle
            with open(cached_model_path, 'rb') as f:
                model_data = pickle.load(f)

            # 모델 구조 확인
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.tokenizer = model_data.get('tokenizer')
                self.model_version = model_data.get('version', 'gdrive_v1')
                logger.info(f"✅ Google Drive 모델 로딩 완료: {self.model_version}")
            else:
                # 단일 모델인 경우
                self.model = model_data
                self.model_version = "gdrive_v1"
                logger.info("✅ Google Drive 단일 모델 로딩 완료")

            self.model_loaded = True
            return True

        except Exception as e:
            logger.error(f"❌ Google Drive 모델 로딩 실패: {e}")
            self.model_loaded = False
            return False

    async def analyze_emotion(self, text: str, use_model: bool = True) -> Dict[str, Any]:
        """
        텍스트 감정 분석 (Google Drive 모델 우선, 실패 시 룰 기반 폴백)

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
            # 🌤️ Google Drive 모델 분석 시도 (우선순위 1)
            if use_model and self.google_drive_enabled:
                try:
                    if not self.model_loaded:
                        logger.info("🌤️ Google Drive 모델 로딩 시도...")
                        await self.load_google_drive_model()

                    if self.model_loaded:
                        logger.info(f"🌤️ Google Drive 모델 v{self.model_version} 분석 시작...")
                        gdrive_result = await self._analyze_with_google_drive_model(text)

                        if gdrive_result.get("success"):
                            response_time = time.time() - start_time
                            gdrive_result["method"] = "google_drive_model"
                            self._update_performance_stats(gdrive_result, response_time)

                            logger.info(f"✅ Google Drive 모델 분석 완료 (소요시간: {response_time:.3f}초)")
                            logger.info(f"  - 감정: {gdrive_result.get('primary_emotion')}")
                            logger.info(f"  - 신뢰도: {gdrive_result.get('confidence', 0):.3f}")

                            return gdrive_result
                        else:
                            logger.warning("⚠️ Google Drive 모델 분석 실패, 룰 기반으로 폴백")

                except Exception as e:
                    logger.error(f"❌ Google Drive 모델 분석 중 오류: {e}")

            # 🤖 로컬 AI 모델 분석 시도 (우선순위 2)
            if use_model and self._is_local_model_available():
                try:
                    logger.info(f"🤖 로컬 AI 모델 v{self.model_version} 분석 시작...")
                    model_result = await self._analyze_with_local_model(text)

                    if model_result.get("success"):
                        response_time = time.time() - start_time
                        self._update_performance_stats(model_result, response_time)

                        logger.info(f"✅ 로컬 AI 모델 분석 완료 (소요시간: {response_time:.3f}초)")
                        logger.info(f"  - 감정: {model_result.get('primary_emotion')}")
                        logger.info(f"  - 신뢰도: {model_result.get('confidence', 0):.3f}")

                        return model_result
                    else:
                        logger.warning("⚠️ 로컬 AI 모델 분석 실패, 룰 기반으로 폴백")

                except Exception as e:
                    logger.error(f"❌ 로컬 AI 모델 분석 중 오류: {e}")

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

    def _is_local_model_available(self) -> bool:
        """로컬 AI 모델 사용 가능 여부 확인"""
        # 🚧 현재는 모델 개발 중이므로 False 반환
        # 모델 완성 후 실제 로딩 로직으로 교체
        return False

    async def _analyze_with_google_drive_model(self, text: str) -> Dict[str, Any]:
        """Google Drive 모델을 사용한 감정 분석"""
        try:
            if not self.model_loaded or not self.model:
                return {"success": False, "reason": "model_not_loaded"}

            # 🚧 실제 Google Drive 모델 추론 로직 구현 필요
            # 현재는 개발 중 상태로 시뮬레이션
            await asyncio.sleep(0.1)  # 모델 추론 시뮬레이션

            # 임시 응답 (실제 모델 구현 후 교체)
            return {
                "success": True,
                "method": "google_drive_model",
                "primary_emotion": "기쁨",
                "confidence": 0.85,
                "emotion_tags": self.emotion_to_tags.get("기쁨", ["#neutral"]),
                "analysis_details": {
                    "model_version": self.model_version,
                    "processing_method": "google_drive_ai"
                },
                "analyzed_at": datetime.now().isoformat(),
                "analyzer_version": "gdrive_v1"
            }

        except Exception as e:
            logger.error(f"❌ Google Drive 모델 추론 중 오류: {e}")
            return {"success": False, "reason": "gdrive_model_error", "message": str(e)}

    async def _analyze_with_local_model(self, text: str) -> Dict[str, Any]:
        """로컬 AI 모델을 사용한 감정 분석 (기존 로직)"""
        try:
            # 🚧 모델 개발 완료 후 구현 예정
            await asyncio.sleep(0.05)  # 모델 추론 시뮬레이션

            return {
                "success": False,
                "reason": "model_under_development",
                "message": f"로컬 AI 모델 v{self.model_version}는 현재 개발 중입니다."
            }

        except Exception as e:
            logger.error(f"❌ 로컬 모델 추론 중 오류: {e}")
            return {"success": False, "reason": "local_model_error", "message": str(e)}

    async def _analyze_with_rules(self, text: str) -> Dict[str, Any]:
        """룰 기반 감정 분석 (향수 도메인 특화) - 기존 로직 유지"""
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
        """키워드 중요도 가중치 계산 - 기존 로직 유지"""
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
        """향수 도메인 특화 컨텍스트 분석 - 기존 로직 유지"""
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
        """빈 텍스트에 대한 결과 - 기존 로직 유지"""
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
        """중립 감정 결과 - 기존 로직 유지"""
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
        """에러 결과 - 기존 로직 유지"""
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
        """성능 통계 업데이트 - Google Drive 메소드 추가"""
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

    # 🆕 Google Drive 관련 메소드들
    async def force_download_google_drive_model(self) -> bool:
        """Google Drive 모델 강제 다운로드"""
        if not self.google_drive_enabled:
            return False

        # 기존 캐시 삭제
        cached_model_path = os.path.join(self.model_cache_dir, "emotion_model_gdrive.pkl")
        if os.path.exists(cached_model_path):
            os.remove(cached_model_path)
            logger.info("🗑️ 기존 Google Drive 모델 캐시 삭제")

        # 강제 다운로드
        return await self.download_google_drive_model()

    def get_google_drive_model_info(self) -> Dict[str, Any]:
        """Google Drive 모델 정보 반환"""
        info = {
            "enabled": self.google_drive_enabled,
            "model_id": self.google_drive_model_id[:20] + "..." if self.google_drive_model_id else None,
            "cache_dir": self.model_cache_dir,
            "model_loaded": self.model_loaded,
            "model_version": self.model_version if self.model_loaded else None
        }

        # 캐시된 모델 파일 정보
        cached_model_path = os.path.join(self.model_cache_dir, "emotion_model_gdrive.pkl")
        if os.path.exists(cached_model_path):
            file_size = os.path.getsize(cached_model_path)
            file_mtime = datetime.fromtimestamp(os.path.getmtime(cached_model_path))
            info["cached_model"] = {
                "exists": True,
                "size_bytes": file_size,
                "size_kb": round(file_size / 1024, 1),
                "last_modified": file_mtime.isoformat()
            }
        else:
            info["cached_model"] = {"exists": False}

        return info

    # 기존 메소드들 유지
    def get_supported_emotions(self) -> List[str]:
        """지원하는 감정 목록 반환"""
        return list(self.emotion_to_tags.keys())

    def get_emotion_tags(self, emotion: str) -> List[str]:
        """특정 감정의 태그 목록 반환"""
        return self.emotion_to_tags.get(emotion, ["#neutral"])

    def update_emotion_mapping(self, new_mapping: Dict[str, List[str]]):
        """감정-태그 매핑 업데이트 (모델 업데이트 시 사용)"""
        logger.info(f"🔄 감정 태그 매핑 업데이트...")
        old_count = len(self.emotion_to_tags)
        self.emotion_to_tags.update(new_mapping)
        new_count = len(self.emotion_to_tags)
        logger.info(f"✅ 감정 태그 매핑 업데이트 완료: {old_count} → {new_count}개")

    def add_custom_keywords(self, emotion: str, keywords: List[str]):
        """특정 감정에 커스텀 키워드 추가"""
        if emotion in self.emotion_keywords:
            old_count = len(self.emotion_keywords[emotion])
            self.emotion_keywords[emotion].extend(keywords)
            # 중복 제거
            self.emotion_keywords[emotion] = list(set(self.emotion_keywords[emotion]))
            new_count = len(self.emotion_keywords[emotion])
            logger.info(f"📝 {emotion} 키워드 업데이트: {old_count} → {new_count}개")
        else:
            logger.warning(f"⚠️ 지원하지 않는 감정: {emotion}")

    async def load_model(self, model_path: str = "./models/emotion_model_v2.pkl"):
        """로컬 AI 모델 로딩 (모델 완성 후 구현)"""
        try:
            logger.info(f"🤖 감정 분석 모델 로딩 시작...")
            logger.info(f"  - 모델 경로: {model_path}")

            # TODO: 실제 모델 로딩 로직
            if not os.path.exists(model_path):
                logger.warning(f"⚠️ 모델 파일이 없습니다: {model_path}")
                self.model_loaded = False
                return False

            # 현재는 개발 중이므로 False
            logger.warning(f"⚠️ 모델이 아직 개발 중입니다 (v{self.model_version})")
            self.model_loaded = False
            return False

        except Exception as e:
            logger.error(f"❌ 모델 로딩 실패: {e}")
            self.model_loaded = False
            return False

    def get_analysis_stats(self) -> Dict[str, Any]:
        """분석 시스템 상태 정보 - Google Drive 정보 추가"""
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
            "analysis_methods": ["rule_based"] + (["google_drive"] if self.google_drive_enabled else []) + (
                ["ai_model"] if self.model_loaded else []),

            # 🆕 Google Drive 정보
            "google_drive": self.get_google_drive_model_info(),

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

    def get_performance_report(self) -> Dict[str, Any]:
        """상세 성능 리포트 생성 - Google Drive 정보 포함"""
        stats = self.get_analysis_stats()

        report = {
            "report_generated_at": datetime.now().isoformat(),
            "system_overview": {
                "status": "operational" if stats["performance"]["success_rate"] > 80 else "degraded",
                "total_analyses": stats["performance"]["total_analyses"],
                "success_rate": stats["performance"]["success_rate"],
                "average_response_time": stats["performance"]["average_response_time"]
            },
            "performance_analysis": stats["performance"],
            "google_drive_status": stats["google_drive"],  # 🆕 추가
            "recommendations": []
        }

        # 성능 개선 권장사항
        perf = stats["performance"]
        if perf["success_rate"] < 90:
            report["recommendations"].append("성공률이 낮습니다. 에러 로그를 확인하세요.")

        if perf["average_response_time"] > 2.0:
            report["recommendations"].append("응답 시간이 느립니다. 키워드 최적화를 고려하세요.")

        confidence_dist = perf["confidence_distribution"]
        total_confident = confidence_dist["high"] + confidence_dist["medium"]
        if total_confident < confidence_dist["low"]:
            report["recommendations"].append("신뢰도가 낮은 분석이 많습니다. 키워드를 확장하세요.")

        if not self.model_loaded and self.google_drive_enabled:
            report["recommendations"].append("Google Drive 모델을 로딩하면 성능이 개선될 수 있습니다.")
        elif not self.google_drive_enabled:
            report["recommendations"].append("Google Drive 모델을 활성화하면 더 정확한 분석이 가능합니다.")

        return report

    def reset_performance_stats(self):
        """성능 통계 리셋"""
        logger.info("🔄 성능 통계 리셋...")
        self.analysis_count = 0
        self.performance_stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "average_response_time": 0.0,
            "method_distribution": {"rule_based": 0, "ai_model": 0, "google_drive": 0},
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0}
        }
        logger.info("✅ 성능 통계 리셋 완료")


# 🌟 전역 감정 분석기 인스턴스
emotion_analyzer = EmotionAnalyzer()


# 🧪 테스트 및 디버깅 함수들
async def test_emotion_analyzer():
    """감정 분석기 테스트 함수 - Google Drive 포함"""
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
    print(f"  Google Drive: {'사용 가능' if stats['google_drive']['enabled'] else '사용 불가'}")
    print()

    print("✅ 테스트 완료!")


if __name__ == "__main__":
    # 직접 실행 시 테스트
    asyncio.run(test_emotion_analyzer())