# utils/email_sender.py - 회원 탈퇴 이메일 기능 추가
import smtplib
import os
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Tuple, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class EmailSender:
    def __init__(self):
        # 환경변수에서 SMTP 설정 가져오기 (기본값 포함)
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_username = os.getenv('SMTP_USERNAME', '')  # 빈 문자열로 초기화
        self.smtp_password = os.getenv('SMTP_PASSWORD', '')  # 빈 문자열로 초기화
        self.from_email = os.getenv('FROM_EMAIL', self.smtp_username)  # FROM_EMAIL이 없으면 username 사용

        # 설정 로깅
        logger.info("📧 이메일 발송자 초기화")
        logger.info(f"  - SMTP 서버: {self.smtp_server}:{self.smtp_port}")
        logger.info(f"  - 발신 이메일: {self.from_email}")
        logger.info(f"  - SMTP 사용자명: {'설정됨' if self.smtp_username else '❌ 없음'}")
        logger.info(f"  - SMTP 비밀번호: {'설정됨' if self.smtp_password else '❌ 없음'}")

    def check_smtp_config(self) -> Tuple[bool, str]:
        """SMTP 설정 유효성 검사"""
        if not self.smtp_username:
            return False, "SMTP_USERNAME 환경변수가 설정되지 않았습니다."

        if not self.smtp_password:
            return False, "SMTP_PASSWORD 환경변수가 설정되지 않았습니다."

        if not self.from_email:
            return False, "FROM_EMAIL 환경변수가 설정되지 않았습니다."

        return True, "SMTP 설정이 올바릅니다."

    def send_verification_email(self, to_email: str, verification_link: str, user_name: str = "사용자") -> Tuple[
        bool, str]:
        """이메일 인증 메일 발송"""
        start_time = datetime.now()
        logger.info(f"📧 이메일 인증 발송 시작")
        logger.info(f"  - 수신자: {to_email}")
        logger.info(f"  - 사용자명: {user_name}")

        try:
            # SMTP 설정 검증
            config_valid, config_message = self.check_smtp_config()
            if not config_valid:
                logger.error(f"❌ SMTP 설정 오류: {config_message}")
                return False, config_message

            # 이메일 내용 구성
            subject = "Whiff - 이메일 인증을 완료해주세요"

            html_body = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>이메일 인증</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                    .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                    .header {{ color: #4CAF50; text-align: center; }}
                    .button {{ 
                        background-color: #4CAF50; 
                        color: white; 
                        padding: 12px 30px; 
                        text-decoration: none; 
                        border-radius: 5px; 
                        display: inline-block;
                        font-weight: bold;
                        margin: 20px 0;
                    }}
                    .footer {{ font-size: 12px; color: #666; margin-top: 30px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2 class="header">🌸 Whiff 이메일 인증</h2>
                    <p>안녕하세요, <strong>{user_name}</strong>님!</p>
                    <p>Whiff에 가입해주셔서 감사합니다. 아래 버튼을 클릭하여 이메일 인증을 완료해주세요.</p>
                    <div style="text-align: center;">
                        <a href="{verification_link}" class="button">✅ 이메일 인증하기</a>
                    </div>
                    <div class="footer">
                        <hr>
                        <p>이 이메일을 요청하지 않으셨다면 무시해주세요.</p>
                        <p><strong>Whiff 팀</strong> 드림 🌹</p>
                    </div>
                </div>
            </body>
            </html>
            """

            text_body = f"""
Whiff 이메일 인증

안녕하세요, {user_name}님!

Whiff에 가입해주셔서 감사합니다. 
아래 링크를 클릭하여 이메일 인증을 완료해주세요.

{verification_link}

이 이메일을 요청하지 않으셨다면 무시해주세요.

Whiff 팀 드림
            """

            # 이메일 발송
            success, message = self._send_email(to_email, subject, html_body, text_body)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            if success:
                logger.info(f"✅ 이메일 발송 성공! (소요시간: {duration:.2f}초)")
                return True, f"이메일이 성공적으로 발송되었습니다."
            else:
                logger.error(f"❌ 이메일 발송 실패: {message}")
                return False, f"이메일 발송 실패: {message}"

        except Exception as e:
            error_msg = f"이메일 발송 중 예외 발생: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

    def send_withdrawal_confirmation_email(
            self,
            to_email: str,
            user_name: str = "사용자",
            deleted_data: Optional[Dict] = None,
            reason: Optional[str] = None
    ) -> Tuple[bool, str]:
        """회원 탈퇴 확인 이메일 발송"""
        start_time = datetime.now()
        logger.info(f"📧 회원 탈퇴 확인 이메일 발송 시작")
        logger.info(f"  - 수신자: {to_email}")
        logger.info(f"  - 사용자명: {user_name}")

        try:
            # SMTP 설정 검증
            config_valid, config_message = self.check_smtp_config()
            if not config_valid:
                logger.error(f"❌ SMTP 설정 오류: {config_message}")
                return False, config_message

            # 삭제된 데이터 정보 구성
            deleted_info = ""
            if deleted_data:
                total_items = sum(deleted_data.values())
                deleted_info = f"""
                <h3>삭제된 데이터 정보:</h3>
                <ul>
                    <li>사용자 프로필: {deleted_data.get('user_profile', 0)}건</li>
                    <li>시향 일기: {deleted_data.get('diaries', 0)}건</li>
                    <li>추천 기록: {deleted_data.get('recommendations', 0)}건</li>
                    <li>임시 데이터: {deleted_data.get('temp_users', 0)}건</li>
                </ul>
                <p><strong>총 {total_items}건의 데이터가 영구적으로 삭제되었습니다.</strong></p>
                """

            # 탈퇴 사유 정보
            reason_info = ""
            if reason:
                reason_info = f"<p><strong>탈퇴 사유:</strong> {reason}</p>"

            # 이메일 내용 구성
            subject = "Whiff - 회원 탈퇴가 완료되었습니다"

            html_body = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>회원 탈퇴 완료</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                    .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                    .header {{ color: #ff6b6b; text-align: center; }}
                    .info-box {{ 
                        background-color: #f8f9fa; 
                        border: 1px solid #dee2e6; 
                        border-radius: 5px; 
                        padding: 15px; 
                        margin: 20px 0; 
                    }}
                    .footer {{ font-size: 12px; color: #666; margin-top: 30px; }}
                    ul {{ padding-left: 20px; }}
                    li {{ margin: 5px 0; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2 class="header">👋 Whiff 회원 탈퇴 완료</h2>
                    <p>안녕하세요, <strong>{user_name}</strong>님</p>
                    <p>Whiff 서비스 회원 탈퇴가 정상적으로 완료되었습니다.</p>

                    <div class="info-box">
                        <p><strong>탈퇴 일시:</strong> {datetime.now().strftime('%Y년 %m월 %d일 %H:%M')}</p>
                        {reason_info}
                        {deleted_info}
                    </div>

                    <p>그동안 Whiff를 이용해주셔서 진심으로 감사했습니다.</p>
                    <p>더 나은 서비스로 언젠가 다시 만날 수 있기를 기대합니다.</p>

                    <div class="info-box">
                        <h4>🔒 개인정보 처리 안내</h4>
                        <ul>
                            <li>모든 개인 데이터가 영구적으로 삭제되었습니다</li>
                            <li>이 작업은 되돌릴 수 없습니다</li>
                            <li>법정 보관 의무가 있는 데이터는 관련 법령에 따라 처리됩니다</li>
                        </ul>
                    </div>

                    <div class="footer">
                        <hr>
                        <p>문의사항이 있으시면 고객센터로 연락해주세요.</p>
                        <p><strong>Whiff 팀</strong> 드림</p>
                    </div>
                </div>
            </body>
            </html>
            """

            # 텍스트 버전
            deleted_text = ""
            if deleted_data:
                total_items = sum(deleted_data.values())
                deleted_text = f"""
삭제된 데이터 정보:
- 사용자 프로필: {deleted_data.get('user_profile', 0)}건
- 시향 일기: {deleted_data.get('diaries', 0)}건
- 추천 기록: {deleted_data.get('recommendations', 0)}건
- 임시 데이터: {deleted_data.get('temp_users', 0)}건

총 {total_items}건의 데이터가 영구적으로 삭제되었습니다.
"""

            reason_text = f"탈퇴 사유: {reason}\n" if reason else ""

            text_body = f"""
Whiff 회원 탈퇴 완료

안녕하세요, {user_name}님

Whiff 서비스 회원 탈퇴가 정상적으로 완료되었습니다.

탈퇴 일시: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M')}
{reason_text}
{deleted_text}

그동안 Whiff를 이용해주셔서 진심으로 감사했습니다.
더 나은 서비스로 언젠가 다시 만날 수 있기를 기대합니다.

개인정보 처리 안내:
- 모든 개인 데이터가 영구적으로 삭제되었습니다
- 이 작업은 되돌릴 수 없습니다
- 법정 보관 의무가 있는 데이터는 관련 법령에 따라 처리됩니다

문의사항이 있으시면 고객센터로 연락해주세요.

Whiff 팀 드림
            """

            # 이메일 발송
            success, message = self._send_email(to_email, subject, html_body, text_body)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            if success:
                logger.info(f"✅ 회원 탈퇴 확인 이메일 발송 성공! (소요시간: {duration:.2f}초)")
                return True, f"회원 탈퇴 확인 이메일이 성공적으로 발송되었습니다."
            else:
                logger.error(f"❌ 회원 탈퇴 확인 이메일 발송 실패: {message}")
                return False, f"회원 탈퇴 확인 이메일 발송 실패: {message}"

        except Exception as e:
            error_msg = f"회원 탈퇴 확인 이메일 발송 중 예외 발생: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

    def send_password_reset_email(
            self,
            to_email: str,
            reset_link: str,
            user_name: str = "사용자"
    ) -> Tuple[bool, str]:
        """비밀번호 재설정 이메일 발송"""
        start_time = datetime.now()
        logger.info(f"📧 비밀번호 재설정 이메일 발송 시작")
        logger.info(f"  - 수신자: {to_email}")
        logger.info(f"  - 사용자명: {user_name}")

        try:
            # SMTP 설정 검증
            config_valid, config_message = self.check_smtp_config()
            if not config_valid:
                logger.error(f"❌ SMTP 설정 오류: {config_message}")
                return False, config_message

            # 이메일 내용 구성
            subject = "Whiff - 비밀번호 재설정 요청"

            html_body = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>비밀번호 재설정</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                    .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                    .header {{ color: #2196F3; text-align: center; }}
                    .button {{ 
                        background-color: #2196F3; 
                        color: white; 
                        padding: 12px 30px; 
                        text-decoration: none; 
                        border-radius: 5px; 
                        display: inline-block;
                        font-weight: bold;
                        margin: 20px 0;
                    }}
                    .warning {{ 
                        background-color: #fff3cd; 
                        border: 1px solid #ffeaa7; 
                        border-radius: 5px; 
                        padding: 15px; 
                        margin: 20px 0; 
                    }}
                    .footer {{ font-size: 12px; color: #666; margin-top: 30px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2 class="header">🔑 Whiff 비밀번호 재설정</h2>
                    <p>안녕하세요, <strong>{user_name}</strong>님!</p>
                    <p>Whiff 계정의 비밀번호 재설정을 요청하셨습니다.</p>
                    <p>아래 버튼을 클릭하여 새로운 비밀번호를 설정해주세요.</p>

                    <div style="text-align: center;">
                        <a href="{reset_link}" class="button">🔒 비밀번호 재설정하기</a>
                    </div>

                    <div class="warning">
                        <h4>⚠️ 보안 안내</h4>
                        <ul>
                            <li>이 링크는 보안을 위해 24시간 후 만료됩니다</li>
                            <li>비밀번호 재설정을 요청하지 않으셨다면 이 이메일을 무시해주세요</li>
                            <li>의심스러운 활동이 있다면 즉시 고객센터로 연락해주세요</li>
                        </ul>
                    </div>

                    <div class="footer">
                        <hr>
                        <p>이 이메일을 요청하지 않으셨다면 무시해주세요.</p>
                        <p><strong>Whiff 팀</strong> 드림 🌹</p>
                    </div>
                </div>
            </body>
            </html>
            """

            text_body = f"""
Whiff 비밀번호 재설정

안녕하세요, {user_name}님!

Whiff 계정의 비밀번호 재설정을 요청하셨습니다.
아래 링크를 클릭하여 새로운 비밀번호를 설정해주세요.

{reset_link}

보안 안내:
- 이 링크는 보안을 위해 24시간 후 만료됩니다
- 비밀번호 재설정을 요청하지 않으셨다면 이 이메일을 무시해주세요
- 의심스러운 활동이 있다면 즉시 고객센터로 연락해주세요

이 이메일을 요청하지 않으셨다면 무시해주세요.

Whiff 팀 드림
            """

            # 이메일 발송
            success, message = self._send_email(to_email, subject, html_body, text_body)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            if success:
                logger.info(f"✅ 비밀번호 재설정 이메일 발송 성공! (소요시간: {duration:.2f}초)")
                return True, f"비밀번호 재설정 이메일이 성공적으로 발송되었습니다."
            else:
                logger.error(f"❌ 비밀번호 재설정 이메일 발송 실패: {message}")
                return False, f"비밀번호 재설정 이메일 발송 실패: {message}"

        except Exception as e:
            error_msg = f"비밀번호 재설정 이메일 발송 중 예외 발생: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

    def _send_email(self, to_email: str, subject: str, html_body: str, text_body: str) -> Tuple[bool, str]:
        """실제 이메일 발송"""
        try:
            logger.info(f"🔌 SMTP 서버 연결 시도: {self.smtp_server}:{self.smtp_port}")

            # 메시지 구성
            msg = MIMEMultipart('alternative')
            msg['From'] = self.from_email
            msg['To'] = to_email
            msg['Subject'] = subject

            # 텍스트 및 HTML 파트 추가
            text_part = MIMEText(text_body, 'plain', 'utf-8')
            html_part = MIMEText(html_body, 'html', 'utf-8')

            msg.attach(text_part)
            msg.attach(html_part)

            # SMTP 서버 연결 및 발송
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)

                logger.info(f"✅ 이메일 전송 완료")
                return True, "발송 성공"

        except smtplib.SMTPAuthenticationError as e:
            error_msg = f"SMTP 인증 실패: Gmail 앱 비밀번호를 확인하세요."
            logger.error(f"❌ {error_msg}")
            return False, error_msg

        except Exception as e:
            error_msg = f"SMTP 발송 중 예외: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

    def test_smtp_connection(self) -> Tuple[bool, str]:
        """SMTP 연결 테스트"""
        try:
            config_valid, config_message = self.check_smtp_config()
            if not config_valid:
                return False, config_message

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                return True, "SMTP 연결 테스트 성공"

        except Exception as e:
            error_msg = f"SMTP 연결 테스트 실패: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg


# 전역 이메일 발송자 인스턴스
email_sender = EmailSender()