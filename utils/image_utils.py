# utils/image_utils.py - 이미지 업로드 및 처리 유틸리티

import os
import uuid
import logging
from typing import Tuple, Optional
from PIL import Image, ImageOps
from fastapi import UploadFile, HTTPException
import aiofiles

logger = logging.getLogger("image_utils")

# 📂 이미지 저장 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "../uploads/diary_images")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_WIDTH = 1920
MAX_HEIGHT = 1920
THUMBNAIL_SIZE = (400, 400)


def create_upload_directories():
    """업로드 디렉토리 생성"""
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        os.makedirs(os.path.join(UPLOAD_DIR, "thumbnails"), exist_ok=True)
        logger.info(f"✅ 업로드 디렉토리 생성: {UPLOAD_DIR}")
    except Exception as e:
        logger.error(f"❌ 업로드 디렉토리 생성 실패: {e}")
        raise


def validate_image_file(file: UploadFile) -> Tuple[bool, str]:
    """
    이미지 파일 검증

    Returns:
        Tuple[bool, str]: (유효성, 메시지)
    """
    try:
        # 1. 파일 크기 검증
        if hasattr(file, 'size') and file.size > MAX_FILE_SIZE:
            return False, f"파일 크기가 너무 큽니다. 최대 {MAX_FILE_SIZE // (1024 * 1024)}MB까지 가능합니다."

        # 2. 파일 확장자 검증
        file_extension = os.path.splitext(file.filename.lower())[1] if file.filename else ""
        if file_extension not in ALLOWED_EXTENSIONS:
            return False, f"지원하지 않는 파일 형식입니다. {', '.join(ALLOWED_EXTENSIONS)} 파일만 업로드 가능합니다."

        # 3. MIME 타입 검증
        if not file.content_type or not file.content_type.startswith('image/'):
            return False, "이미지 파일만 업로드 가능합니다."

        return True, "유효한 이미지 파일입니다."

    except Exception as e:
        logger.error(f"❌ 이미지 파일 검증 오류: {e}")
        return False, f"파일 검증 중 오류가 발생했습니다: {str(e)}"


def generate_unique_filename(original_filename: str, user_id: str) -> str:
    """고유한 파일명 생성"""
    try:
        # 파일 확장자 추출
        file_extension = os.path.splitext(original_filename.lower())[1] if original_filename else ".jpg"

        # UUID + 사용자ID + 확장자로 고유 파일명 생성
        unique_id = str(uuid.uuid4())
        safe_user_id = "".join(c for c in user_id if c.isalnum())[:10]  # 안전한 사용자ID (10자 제한)

        filename = f"{safe_user_id}_{unique_id}{file_extension}"
        return filename

    except Exception as e:
        logger.error(f"❌ 파일명 생성 오류: {e}")
        # 폴백: UUID만 사용
        return f"{uuid.uuid4()}.jpg"


async def save_uploaded_image(file: UploadFile, user_id: str) -> Tuple[bool, str, Optional[dict]]:
    """
    업로드된 이미지 저장 및 처리

    Returns:
        Tuple[bool, str, Optional[dict]]: (성공여부, 파일경로/오류메시지, 메타데이터)
    """
    try:
        # 1. 디렉토리 생성
        create_upload_directories()

        # 2. 파일 검증
        is_valid, message = validate_image_file(file)
        if not is_valid:
            return False, message, None

        # 3. 고유 파일명 생성
        filename = generate_unique_filename(file.filename, user_id)
        file_path = os.path.join(UPLOAD_DIR, filename)
        thumbnail_path = os.path.join(UPLOAD_DIR, "thumbnails", f"thumb_{filename}")

        # 4. 원본 파일 저장
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        logger.info(f"✅ 원본 이미지 저장: {filename}")

        # 5. 이미지 처리 (리사이징 및 썸네일 생성)
        try:
            with Image.open(file_path) as img:
                # EXIF 정보 기반 회전 보정
                img = ImageOps.exif_transpose(img)

                # 원본 이미지 메타데이터
                original_width, original_height = img.size
                file_size = os.path.getsize(file_path)

                # 원본 이미지 리사이징 (너무 큰 경우)
                if original_width > MAX_WIDTH or original_height > MAX_HEIGHT:
                    img.thumbnail((MAX_WIDTH, MAX_HEIGHT), Image.Resampling.LANCZOS)
                    img.save(file_path, optimize=True, quality=85)
                    logger.info(f"🔄 이미지 리사이징 완료: {img.size}")

                # 썸네일 생성
                thumbnail_img = img.copy()
                thumbnail_img.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
                thumbnail_img.save(thumbnail_path, optimize=True, quality=80)
                logger.info(f"🖼️ 썸네일 생성 완료: {thumbnail_img.size}")

                # 메타데이터 반환
                metadata = {
                    "filename": filename,
                    "original_filename": file.filename,
                    "file_size": file_size,
                    "original_size": (original_width, original_height),
                    "processed_size": img.size,
                    "thumbnail_size": thumbnail_img.size,
                    "content_type": file.content_type,
                    "user_id": user_id
                }

                return True, filename, metadata

        except Exception as e:
            logger.error(f"❌ 이미지 처리 오류: {e}")
            # 처리 실패해도 원본 파일은 유지
            return True, filename, {
                "filename": filename,
                "original_filename": file.filename,
                "file_size": os.path.getsize(file_path),
                "processing_error": str(e),
                "user_id": user_id
            }

    except Exception as e:
        logger.error(f"❌ 이미지 저장 실패: {e}")
        return False, f"이미지 저장 중 오류가 발생했습니다: {str(e)}", None


def get_image_url(filename: str, request_base_url: str) -> str:
    """이미지 URL 생성"""
    try:
        # 프로덕션 환경에서는 실제 도메인 사용
        base_url = request_base_url.rstrip('/')
        image_url = f"{base_url}/uploads/diary_images/{filename}"
        return image_url
    except Exception as e:
        logger.error(f"❌ 이미지 URL 생성 오류: {e}")
        return ""


def get_thumbnail_url(filename: str, request_base_url: str) -> str:
    """썸네일 URL 생성"""
    try:
        base_url = request_base_url.rstrip('/')
        thumbnail_filename = f"thumb_{filename}"
        thumbnail_url = f"{base_url}/uploads/diary_images/thumbnails/{thumbnail_filename}"
        return thumbnail_url
    except Exception as e:
        logger.error(f"❌ 썸네일 URL 생성 오류: {e}")
        return ""


def delete_image_files(filename: str) -> bool:
    """이미지 파일 삭제 (원본 + 썸네일)"""
    try:
        success = True

        # 원본 파일 삭제
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"🗑️ 원본 이미지 삭제: {filename}")

        # 썸네일 파일 삭제
        thumbnail_path = os.path.join(UPLOAD_DIR, "thumbnails", f"thumb_{filename}")
        if os.path.exists(thumbnail_path):
            os.remove(thumbnail_path)
            logger.info(f"🗑️ 썸네일 삭제: thumb_{filename}")

        return success

    except Exception as e:
        logger.error(f"❌ 이미지 파일 삭제 실패: {e}")
        return False


def get_upload_stats() -> dict:
    """업로드 통계 정보"""
    try:
        if not os.path.exists(UPLOAD_DIR):
            return {"total_files": 0, "total_size": 0}

        total_files = 0
        total_size = 0

        for root, dirs, files in os.walk(UPLOAD_DIR):
            for file in files:
                if not file.startswith('.'):  # 숨김 파일 제외
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        total_files += 1
                        total_size += os.path.getsize(file_path)

        return {
            "total_files": total_files,
            "total_size": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "upload_dir": UPLOAD_DIR
        }

    except Exception as e:
        logger.error(f"❌ 업로드 통계 조회 오류: {e}")
        return {"error": str(e)}