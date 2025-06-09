# utils/emotion_analyzer.py - Whiff 감정 분석 시스템 (Google Drive 연동 버전)

import re
import logging
import json
import os
import asyncio
import time
import pickle
import requests
from typing import List, Dict, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
from collections import Counter
import numpy as np

# Google Drive API 관련 임포트
try:
    from googleapiclient.discovery import build
    from google.oauth2.service_account import Credentials
    from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
    import io

    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False
    logging.warning("Google Drive API 라이브러리가 설치되지 않았습니다. pip install google-api-python-client google-auth 실행하세요.")

logger = logging.getLogger(__name__)


class GoogleDriveManager:
    """Google Drive 연동 관리자"""

    def __init__(self, credentials_path: Optional[str] = None):
        """Google Drive 매니저 초기화"""
        self.service = None
        self.credentials_path = credentials_path or os.getenv('GOOGLE_DRIVE_CREDENTIALS_PATH')
        self.is_connected = False

        if GOOGLE_DRIVE_AVAILABLE and self.credentials_path:
            self._initialize_drive_service()

    def _initialize_drive_service(self):
        """Google Drive 서비스 초기화"""
        try:
            if not os.path.exists(self.credentials_path):
                logger.error(f"Google Drive 인증 파일이 없습니다: {self.credentials_path}")
                return False

            # 서비스 계정 인증
            scopes = ['https://www.googleapis.com/auth/drive.readonly',
                      'https://www.googleapis.com/auth/drive.file']

            credentials = Credentials.from_service_account_file(
                self.credentials_path, scopes=scopes
            )

            self.service = build('drive', 'v3', credentials=credentials)
            self.is_connected = True

            logger.info("✅ Google Drive 서비스 초기화 완료")
            return True

        except Exception as e:
            logger.error(f"❌ Google Drive 서비스 초기화 실패: {e}")
            return False

    async def download_file_content(self, file_id: str) -> Optional[str]:
        """Google Drive 파일 내용 다운로드"""
        if not self.is_connected:
            logger.warning("Google Drive에 연결되지 않았습니다.")
            return None

        try:
            # 파일 메타데이터 가져오기
            file_metadata = self.service.files().get(fileId=file_id).execute()
            file_name = file_metadata.get('name', 'unknown')

            logger.info(f"📁 Google Drive 파일 다운로드 시작: {file_name}")

            # 파일 내용 다운로드
            request = self.service.files().get_media(fileId=file_id)
            file_content = io.BytesIO()
            downloader = MediaIoBaseDownload(file_content, request)

            done = False
            while done is False:
                status, done = downloader.next_chunk()

            # 텍스트로 디코딩
            file_content.seek(0)
            content = file_content.read().decode('utf-8')

            logger.info(f"✅ 파일 다운로드 완료: {file_name} ({len(content)}자)")
            return content

        except Exception as e:
            logger.error(f"❌ Google Drive 파일 다운로드 실패: {e}")
            return None

    async def upload_analysis_result(self, content: str, filename: str, folder_id: Optional[str] = None) -> Optional[
        str]:
        """감정 분석 결과를 Google Drive에 업로드"""
        if not self.is_connected:
            logger.warning("Google Drive에 연결되지 않았습니다.")
            return None

        try:
            # 파일 메타데이터 설정
            file_metadata = {
                'name': filename,
                'parents': [folder_id] if folder_id else []
            }

            # 파일 내용을 BytesIO로 변환
            media = MediaIoBaseUpload(
                io.BytesIO(content.encode('utf-8')),
                mimetype='text/plain'
            )

            # 파일 업로드
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()

            file_id = file.get('id')
            logger.info(f"✅ 분석 결과 업로드 완료: {filename} (ID: {file_id})")

            return file_id

        except Exception as e:
            logger.error(f"❌ Google Drive 업로드 실패: {e}")
            return None

    async def list_analysis_files(self, folder_id: Optional[str] = None) -> List[Dict[str, str]]:
        """분석 파일 목록 조회"""
        if not self.is_connected:
            return []

        try:
            query = "mimeType='text/plain'"
            if folder_id:
                query += f" and '{folder_id}' in parents"

            results = self.service.files().list(
                q=query,
                pageSize=50,
                fields="nextPageToken, files(id, name, modifiedTime, size)"
            ).execute()

            files = results.get('files', [])

            logger.info(f"📁 Google Drive에서 {len(files)}개 분석 파일 발견")

            return files

        except Exception as e:
            logger.error(f"❌ Google Drive 파일 목록 조회 실패: {e}")
            return []


class EmotionAnalyzer:
    """
    Whiff 시향일기 텍스트의 감정 분석 및 태그 생성 서비스 (Google Drive 연동)

    Features:
    - 룰 기반 감정 분석 + AI 모델 준비
    - Google Drive 연동 (분석 결과 저장/로드)
    - 8개 핵심 감정 지원
    - 향수 도메인 특화 키워드
    - 실시간 학습 데이터 수집
    - 성능 모니터링 및 Google Drive 백업
    """

    def __init__(self, google_drive_credentials: Optional[str] = None):
        """감정 분석기 초기화 (Google Drive 연동)"""
        logger.info("🎭 감정 분석기 초기화 시작... (Google Drive 연동)")

        # Google Drive 매니저 초기화
        self.gdrive_manager = GoogleDriveManager(google_drive_credentials)

        # 🎯 감정별 태그 매핑 (Google Drive에서 동기화 가능)
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
                "불안", "걱정", "긴장", "떨려", "두려운", "무서운", "조마조마",
                "어색", "부담", "압박", "스트레스", "불편", "어색해",
                "이상해", "어색한", "답답", "무거운"
            ],
            "당황": [
                "당황", "놀란", "혼란", "어리둥절", "멍한", "모르겠다", "헷갈려",
                "이상", "예상과 달라", "의외", "신기", "특이", "독특",
                "예상못한", "뜻밖의", "갑작스러운"
            ],
            "분노": [
                "화가", "짜증", "열받", "분노", "격정", "싫어", "별로", "최악",
                "자극적", "강렬", "과해", "부담스러워", "독해", "역겨운",
                "끔찍", "못참겠", "견딜수없", "극혐"
            ],
            "상처": [
                "상처", "아픈", "서운", "실망", "아쉬워", "슬픈", "힘든",
                "섭섭", "마음아픈", "쓸쓸", "그리운", "그립", "애틋", "안타까운", "아련한"
            ],
            "슬픔": [
                "슬퍼", "눈물", "애절", "처량", "고독", "외로운", "쓸쓸",
                "먹먹", "찡한", "울컥", "진한", "깊은", "차가운", "무거운", "침울한", "암울한"
            ],
            "우울": [
                "우울", "답답", "무기력", "절망", "어둠", "침울", "멜랑콜리",
                "블루", "그늘진", "어두운", "막막한", "절망적", "희망없는", "의욕없는", "공허한"
            ],
            "흥분": [
                "흥분", "신나", "두근", "설렘", "활기", "생동감", "에너지",
                "활발", "역동적", "펄떡", "톡톡", "팡팡", "생생한", "활력", "젊은", "발랄한"
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
        self.model_version = "v2_gdrive"  # Google Drive 연동 버전
        self.analysis_count = 0

        # 📊 성능 통계 (Google Drive 백업)
        self.performance_stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "average_response_time": 0.0,
            "method_distribution": {"rule_based": 0, "ai_model": 0},
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
            "gdrive_operations": {"uploads": 0, "downloads": 0, "sync_failures": 0}
        }

        # 🔄 학습 데이터 수집 (Google Drive 동기화)
        self.learning_data = []
        self.last_gdrive_sync = None
        self.sync_interval = timedelta(hours=1)  # 1시간마다 동기화

        # 🚀 초기화 완료
        logger.info("✅ 감정 분석기 초기화 완료 (Google Drive 연동)")
        logger.info(f"  - 지원 감정: {list(self.emotion_to_tags.keys())}")
        logger.info(f"  - 모델 버전: {self.model_version}")
        logger.info(f"  - Google Drive 연결: {'✅' if self.gdrive_manager.is_connected else '❌'}")
        logger.info(f"  - 총 키워드: {sum(len(keywords) for keywords in self.emotion_keywords.values())}개")

    async def analyze_emotion(self, text: str, use_model: bool = True, save_to_gdrive: bool = False) -> Dict[str, Any]:
        """
        텍스트 감정 분석 (Google Drive 연동)

        Args:
            text: 분석할 텍스트
            use_model: AI 모델 사용 여부
            save_to_gdrive: Google Drive에 결과 저장 여부

        Returns:
            분석 결과 딕셔너리
        """
        start_time = time.time()
        self.analysis_count += 1

        logger.info(f"🎭 감정 분석 시작 (#{self.analysis_count})")
        logger.info(f"  - 텍스트 길이: {len(text)}자")
        logger.info(f"  - 모델 사용: {'✅' if use_model else '❌'}")
        logger.info(f"  - Google Drive 저장: {'✅' if save_to_gdrive else '❌'}")

        # 입력 검증
        if not text or not text.strip():
            return self._create_empty_result()

        if len(text) > 2000:
            logger.warning(f"⚠️ 텍스트가 너무 깁니다: {len(text)}자")
            return self._create_error_result("text_too_long", "텍스트가 2000자를 초과합니다.")

        try:
            # 🤖 AI 모델 분석 시도 (개발 완료 후)
            result = None
            if use_model and self._is_model_available():
                try:
                    logger.info(f"🤖 AI 모델 v{self.model_version} 분석 시작...")
                    result = await self._analyze_with_model(text)

                    if result.get("success"):
                        logger.info(f"✅ AI 모델 분석 완료")
                    else:
                        logger.warning("⚠️ AI 모델 분석 실패, 룰 기반으로 폴백")
                        result = None

                except Exception as e:
                    logger.error(f"❌ AI 모델 분석 중 오류: {e}")
                    result = None

            # 📋 룰 기반 분석 (폴백 또는 기본)
            if result is None:
                logger.info(f"📋 룰 기반 감정 분석 시작...")
                result = await self._analyze_with_rules(text)

            # ⏱️ 응답 시간 계산
            response_time = time.time() - start_time
            result["processing_time"] = round(response_time, 3)

            # 📊 성능 통계 업데이트
            self._update_performance_stats(result, response_time)

            # 📚 학습 데이터 수집
            await self._collect_learning_data(text, result)

            # 💾 Google Drive 저장
            if save_to_gdrive and result.get("success"):
                await self._save_result_to_gdrive(text, result)

            # 🔄 정기 동기화 확인
            await self._check_and_sync_gdrive()

            logger.info(f"✅ 감정 분석 완료 (소요시간: {response_time:.3f}초)")
            logger.info(f"  - 감정: {result.get('primary_emotion')}")
            logger.info(f"  - 신뢰도: {result.get('confidence', 0):.3f}")

            return result

        except Exception as e:
            logger.error(f"❌ 감정 분석 중 예외 발생: {e}")
            return self._create_error_result("analysis_exception", str(e))

    async def sync_with_gdrive(self, force: bool = False) -> Dict[str, Any]:
        """Google Drive와 수동 동기화"""
        if not self.gdrive_manager.is_connected:
            return {"success": False, "message": "Google Drive에 연결되지 않았습니다."}

        try:
            sync_start = time.time()
            logger.info("🔄 Google Drive 동기화 시작...")

            # 1. 감정 키워드 사전 동기화
            keywords_synced = await self._sync_emotion_keywords()

            # 2. 성능 통계 백업
            stats_backed_up = await self._backup_performance_stats()

            # 3. 학습 데이터 동기화
            learning_data_synced = await self._sync_learning_data()

            sync_time = time.time() - sync_start
            self.last_gdrive_sync = datetime.now()

            result = {
                "success": True,
                "sync_time": round(sync_time, 3),
                "operations": {
                    "keywords_synced": keywords_synced,
                    "stats_backed_up": stats_backed_up,
                    "learning_data_synced": learning_data_synced
                },
                "last_sync": self.last_gdrive_sync.isoformat()
            }

            logger.info(f"✅ Google Drive 동기화 완료 (소요시간: {sync_time:.3f}초)")
            return result

        except Exception as e:
            self.performance_stats["gdrive_operations"]["sync_failures"] += 1
            logger.error(f"❌ Google Drive 동기화 실패: {e}")
            return {"success": False, "message": str(e)}

    async def load_emotion_keywords_from_gdrive(self, file_id: str) -> bool:
        """Google Drive에서 감정 키워드 사전 로드"""
        try:
            content = await self.gdrive_manager.download_file_content(file_id)
            if content:
                keywords_data = json.loads(content)
                self.emotion_keywords.update(keywords_data)
                logger.info(f"✅ Google Drive에서 감정 키워드 로드 완료")
                return True
        except Exception as e:
            logger.error(f"❌ Google Drive 키워드 로드 실패: {e}")
        return False

    async def analyze_gdrive_document(self, file_id: str) -> Dict[str, Any]:
        """Google Drive 문서 감정 분석"""
        try:
            # 문서 내용 다운로드
            content = await self.gdrive_manager.download_file_content(file_id)
            if not content:
                return self._create_error_result("gdrive_download_failed", "Google Drive 문서 다운로드 실패")

            # 감정 분석 수행
            result = await self.analyze_emotion(content, save_to_gdrive=True)

            # 문서 정보 추가
            result["source"] = "google_drive"
            result["file_id"] = file_id

            logger.info(f"✅ Google Drive 문서 분석 완료: {file_id}")
            return result

        except Exception as e:
            logger.error(f"❌ Google Drive 문서 분석 실패: {e}")
            return self._create_error_result("gdrive_analysis_failed", str(e))

    async def batch_analyze_gdrive_folder(self, folder_id: str, max_files: int = 10) -> List[Dict[str, Any]]:
        """Google Drive 폴더 내 문서들 일괄 분석"""
        if not self.gdrive_manager.is_connected:
            return []

        try:
            logger.info(f"📁 Google Drive 폴더 일괄 분석 시작: {folder_id}")

            # 폴더 내 파일 목록 조회
            files = await self.gdrive_manager.list_analysis_files(folder_id)
            files = files[:max_files]  # 최대 파일 수 제한

            results = []
            for file_info in files:
                file_id = file_info['id']
                file_name = file_info['name']

                logger.info(f"📄 분석 중: {file_name}")

                # 개별 파일 분석
                analysis_result = await self.analyze_gdrive_document(file_id)
                analysis_result["file_name"] = file_name

                results.append(analysis_result)

                # 과부하 방지를 위한 잠시 대기
                await asyncio.sleep(0.5)

            logger.info(f"✅ 폴더 일괄 분석 완료: {len(results)}개 파일")
            return results

        except Exception as e:
            logger.error(f"❌ Google Drive 폴더 일괄 분석 실패: {e}")
            return []

    # ========================= Private Methods =========================

    def _is_model_available(self) -> bool:
        """AI 모델 사용 가능 여부 확인"""
        return self.model_loaded and self.model is not None

    async def _analyze_with_model(self, text: str) -> Dict[str, Any]:
        """AI 모델을 사용한 감정 분석 (미래 구현)"""
        try:
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
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    matches = len(re.findall(pattern, text_lower))

                    if matches > 0:
                        weight = self._get_keyword_weight(keyword, emotion)
                        score += matches * weight
                        matched_keywords.extend([keyword] * matches)
                        total_keywords_found += matches

                if score > 0:
                    emotion_scores[emotion] = score
                    keyword_matches[emotion] = matched_keywords

            # 🌸 2단계: 향수 도메인 컨텍스트 보정
            context_boost = self._analyze_perfume_context(text_lower, text_words)

            for emotion, boost in context_boost.items():
                if emotion in emotion_scores:
                    emotion_scores[emotion] += boost
                elif boost > 0.5:
                    emotion_scores[emotion] = boost
                    keyword_matches[emotion] = ["context_boost"]

            # 📊 3단계: 결과 계산 및 정규화
            if emotion_scores:
                normalization_factor = max(len(text_words), 1)
                normalized_scores = {}

                for emotion, score in emotion_scores.items():
                    normalized_score = min(score / normalization_factor * 1.5, 1.0)
                    normalized_scores[emotion] = normalized_score

                primary_emotion = max(normalized_scores.keys(), key=lambda x: normalized_scores[x])
                confidence = normalized_scores[primary_emotion]

                confidence_boost = min(total_keywords_found * 0.1, 0.3)
                final_confidence = min(confidence + confidence_boost, 1.0)

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
                    "analyzer_version": "rule_based_gdrive_v1.3"
                }
            else:
                return self._create_neutral_result("no_emotion_keywords")

        except Exception as e:
            logger.error(f"❌ 룰 기반 분석 중 오류: {e}")
            return self._create_error_result("rule_analysis_error", str(e))

    def _get_keyword_weight(self, keyword: str, emotion: str) -> float:
        """키워드 중요도 가중치 계산"""
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
            return 1.5
        elif len(keyword) >= 3:
            return 1.2
        else:
            return 1.0

    def _analyze_perfume_context(self, text_lower: str, text_words: List[str]) -> Dict[str, float]:
        """향수 도메인 특화 컨텍스트 분석"""
        context_boost = {}

        positive_quality_count = sum(text_lower.count(kw) for kw in self.perfume_context_keywords["positive_quality"])
        negative_quality_count = sum(text_lower.count(kw) for kw in self.perfume_context_keywords["negative_quality"])
        intensity_negative_count = sum(
            text_lower.count(kw) for kw in self.perfume_context_keywords["intensity_negative"])

        if positive_quality_count > 0:
            boost_strength = min(positive_quality_count * 0.8, 2.0)
            context_boost["기쁨"] = boost_strength
            if positive_quality_count >= 2:
                context_boost["흥분"] = boost_strength * 0.6

        if negative_quality_count > 0:
            boost_strength = min(negative_quality_count * 0.7, 1.8)
            if negative_quality_count >= 2:
                context_boost["분노"] = boost_strength
            else:
                context_boost["상처"] = boost_strength * 0.8

        if intensity_negative_count > 0:
            boost_strength = min(intensity_negative_count * 0.6, 1.5)
            context_boost["불안"] = boost_strength

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

            method = result.get("method", "unknown")
            if method in self.performance_stats["method_distribution"]:
                self.performance_stats["method_distribution"][method] += 1

            confidence = result.get("confidence", 0.0)
            if confidence >= 0.7:
                self.performance_stats["confidence_distribution"]["high"] += 1
            elif confidence >= 0.4:
                self.performance_stats["confidence_distribution"]["medium"] += 1
            else:
                self.performance_stats["confidence_distribution"]["low"] += 1

        total_time = (self.performance_stats["average_response_time"] *
                      (self.performance_stats["total_analyses"] - 1) + response_time)
        self.performance_stats["average_response_time"] = total_time / self.performance_stats["total_analyses"]

    async def _collect_learning_data(self, text: str, result: Dict[str, Any]):
        """학습 데이터 수집"""
        if result.get("success"):
            learning_item = {
                "timestamp": datetime.now().isoformat(),
                "text": text[:200],  # 개인정보 보호를 위해 텍스트 제한
                "emotion": result.get("primary_emotion"),
                "confidence": result.get("confidence"),
                "method": result.get("method"),
                "processing_time": result.get("processing_time")
            }
            self.learning_data.append(learning_item)

            # 메모리 관리: 최대 1000개 항목 유지
            if len(self.learning_data) > 1000:
                self.learning_data = self.learning_data[-1000:]

    async def _save_result_to_gdrive(self, text: str, result: Dict[str, Any]):
        """분석 결과를 Google Drive에 저장"""
        if not self.gdrive_manager.is_connected:
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"emotion_analysis_{timestamp}.json"

            save_data = {
                "analysis_result": result,
                "original_text": text[:500],  # 텍스트 일부만 저장
                "timestamp": datetime.now().isoformat(),
                "analyzer_version": self.model_version
            }

            content = json.dumps(save_data, ensure_ascii=False, indent=2)

            file_id = await self.gdrive_manager.upload_analysis_result(content, filename)
            if file_id:
                self.performance_stats["gdrive_operations"]["uploads"] += 1
                logger.info(f"💾 분석 결과 Google Drive 저장 완료: {filename}")

        except Exception as e:
            logger.error(f"❌ Google Drive 저장 실패: {e}")

    async def _check_and_sync_gdrive(self):
        """정기 Google Drive 동기화 확인"""
        if not self.gdrive_manager.is_connected:
            return

        now = datetime.now()
        if (self.last_gdrive_sync is None or
                now - self.last_gdrive_sync > self.sync_interval):
            logger.info("🔄 정기 Google Drive 동기화 실행...")
            await self.sync_with_gdrive()

    async def _sync_emotion_keywords(self) -> bool:
        """감정 키워드 사전 Google Drive 동기화"""
        try:
            filename = f"emotion_keywords_{datetime.now().strftime('%Y%m%d')}.json"
            content = json.dumps(self.emotion_keywords, ensure_ascii=False, indent=2)

            file_id = await self.gdrive_manager.upload_analysis_result(content, filename)
            return file_id is not None

        except Exception as e:
            logger.error(f"❌ 키워드 동기화 실패: {e}")
            return False

    async def _backup_performance_stats(self) -> bool:
        """성능 통계 Google Drive 백업"""
        try:
            filename = f"performance_stats_{datetime.now().strftime('%Y%m%d')}.json"
            content = json.dumps(self.performance_stats, ensure_ascii=False, indent=2)

            file_id = await self.gdrive_manager.upload_analysis_result(content, filename)
            return file_id is not None

        except Exception as e:
            logger.error(f"❌ 성능 통계 백업 실패: {e}")
            return False

    async def _sync_learning_data(self) -> bool:
        """학습 데이터 Google Drive 동기화"""
        if not self.learning_data:
            return True

        try:
            filename = f"learning_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
            content = json.dumps(self.learning_data, ensure_ascii=False, indent=2)

            file_id = await self.gdrive_manager.upload_analysis_result(content, filename)
            if file_id:
                self.learning_data.clear()  # 동기화 후 메모리 정리
                return True

        except Exception as e:
            logger.error(f"❌ 학습 데이터 동기화 실패: {e}")

        return False

    # ========================= Public Utility Methods =========================

    def get_supported_emotions(self) -> List[str]:
        """지원하는 감정 목록 반환"""
        return list(self.emotion_to_tags.keys())

    def get_emotion_tags(self, emotion: str) -> List[str]:
        """특정 감정의 태그 목록 반환"""
        return self.emotion_to_tags.get(emotion, ["#neutral"])

    def get_gdrive_status(self) -> Dict[str, Any]:
        """Google Drive 연결 상태 반환"""
        return {
            "connected": self.gdrive_manager.is_connected,
            "last_sync": self.last_gdrive_sync.isoformat() if self.last_gdrive_sync else None,
            "sync_interval_hours": self.sync_interval.total_seconds() / 3600,
            "operations": self.performance_stats["gdrive_operations"],
            "learning_data_count": len(self.learning_data)
        }

    def get_analysis_stats(self) -> Dict[str, Any]:
        """분석 시스템 상태 정보 (Google Drive 포함)"""
        success_rate = 0.0
        if self.performance_stats["total_analyses"] > 0:
            success_rate = (self.performance_stats["successful_analyses"] /
                            self.performance_stats["total_analyses"] * 100)

        return {
            "model_loaded": self.model_loaded,
            "model_version": self.model_version,
            "supported_emotions": len(self.emotion_to_tags),
            "total_keywords": sum(len(keywords) for keywords in self.emotion_keywords.values()),
            "analysis_methods": ["rule_based"] + (["ai_model"] if self.model_loaded else []),

            "performance": {
                "total_analyses": self.performance_stats["total_analyses"],
                "successful_analyses": self.performance_stats["successful_analyses"],
                "success_rate": round(success_rate, 2),
                "average_response_time": round(self.performance_stats["average_response_time"], 3),
                "method_distribution": self.performance_stats["method_distribution"],
                "confidence_distribution": self.performance_stats["confidence_distribution"]
            },

            "google_drive": self.get_gdrive_status(),

            "emotion_list": list(self.emotion_to_tags.keys()),
            "emotion_tags_count": {emotion: len(tags) for emotion, tags in self.emotion_to_tags.items()},

            "system_info": {
                "max_text_length": 2000,
                "supported_languages": ["한국어"],
                "domain_specialization": "향수_리뷰",
                "features": ["Google Drive 연동", "실시간 학습", "성능 모니터링"],
                "last_updated": datetime.now().isoformat()
            }
        }


# 🌟 전역 감정 분석기 인스턴스 (Google Drive 연동)
emotion_analyzer = EmotionAnalyzer(
    google_drive_credentials=os.getenv('GOOGLE_DRIVE_CREDENTIALS_PATH')
)


# 🧪 테스트 함수
async def test_emotion_analyzer_with_gdrive():
    """Google Drive 연동 감정 분석기 테스트"""
    print("🧪 Google Drive 연동 감정 분석기 테스트 시작...\n")

    test_cases = [
        "이 향수 정말 좋아요! 달콤하고 상큼해서 기분이 좋아져요.",
        "향이 너무 진해서 별로예요. 좀 부담스럽네요.",
        "처음 맡았을 때 놀랐어요. 예상과 완전 달라서 당황스러웠어요.",
    ]

    for i, text in enumerate(test_cases, 1):
        print(f"--- 테스트 {i} ---")
        print(f"입력: {text}")

        result = await emotion_analyzer.analyze_emotion(text, save_to_gdrive=True)

        print(f"결과: {result['primary_emotion']} (신뢰도: {result['confidence']:.3f})")
        print(f"태그: {result['emotion_tags']}")
        print(f"방법: {result['method']}")
        print(f"처리시간: {result.get('processing_time', 0):.3f}초")
        print()

    # Google Drive 상태 확인
    gdrive_status = emotion_analyzer.get_gdrive_status()
    print("🔄 Google Drive 상태:")
    print(f"  연결됨: {gdrive_status['connected']}")
    print(f"  마지막 동기화: {gdrive_status['last_sync']}")
    print(f"  업로드 횟수: {gdrive_status['operations']['uploads']}")
    print()

    print("✅ 테스트 완료!")


if __name__ == "__main__":
    asyncio.run(test_emotion_analyzer_with_gdrive())