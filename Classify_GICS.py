import requests
import pandas as pd
from io import BytesIO

def get_gics_classification(trading_day: str) -> pd.DataFrame:
    # 1) 세션 생성 및 User-Agent 설정
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })

    # 2) 실제 GICS 분류 페이지를 먼저 GET 요청 → 세션 쿠키 획득
    landing_url = "https://data.krx.co.kr/contents/MDC/MDI/mdi_dg_industList.jsp"
    session.get(landing_url)

    # 3) OTP 생성
    gen_otp_url = "https://data.krx.co.kr/comm/fileDn/GenerateOTP/generate.cmd"
    gen_otp_payload = {
        "trdDd": trading_day,
        "name": "fileDown",
        "csvxls_isNo": "false",
        "url": "dbms/MDC/MDI/industry/MDCMDI03401"  # 네트워크 탭에서 확인한 서비스 코드
    }
    # 반드시 Referer 헤더에 실제 페이지 주소를 담아 줍니다
    session.headers.update({"Referer": landing_url})
    otp = session.post(gen_otp_url, data=gen_otp_payload).text

    # 4) OTP로 CSV 다운로드
    download_url = "https://data.krx.co.kr/comm/fileDn/download_csv/download.cmd"
    resp = session.post(download_url, data={"code": otp})
    resp.encoding = "euc-kr"  # 한글 인코딩

    # 5) DataFrame 변환
    df = pd.read_csv(BytesIO(resp.content), encoding="euc-kr")
    return df

if __name__ == "__main__":
    # 예시: 2025년 08월 07일 기준 GICS 분류 정보
    gics_df = get_gics_classification("20250807")
    print(gics_df.head())