import yfinance as yf
import pandas as pd
import numpy as np

# 삼성전자 티커 (한국 증시 KRX: 005930.KS)
ticker = "005930.KS"

# 데이터 다운로드 (예: 최근 1년치 일별 데이터)
df = yf.download(ticker, period="1y", interval="1d")

# 인덱스 reset & 컬럼명 소문자 통일
df = df.reset_index()
df.columns = [c.lower() for c in df.columns]
print(df.columns)
# time_idx 컬럼 추가 (0부터 시작하는 정수형 시계열 인덱스)
df["time_idx"] = np.arange(len(df))

# ticker 컬럼 추가 (모델 그룹 아이디용)
df["ticker"] = ticker

# 기술지표 예시 (단순 이동평균 SMA 5, SMA 10)
df["tech_indicator_1"] = df["close"].rolling(window=5).mean().fillna(method='bfill')
df["tech_indicator_2"] = df["close"].rolling(window=10).mean().fillna(method='bfill')

# TFTStockModel에서 사용할 컬럼만 선택 및 순서 맞추기
tft_df = df[[
    "time_idx", "ticker", "open", "high", "low", "close", "volume", "tech_indicator_1", "tech_indicator_2"
]]

print(tft_df.head())