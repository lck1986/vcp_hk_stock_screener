# -*- coding: utf-8 -*-
"""
Hong Kong VCP Screener – Enhanced Edition
- Covers Hang Seng Index (HSI), Hang Seng TECH, Hang Seng China Enterprises (HSCEI)
- Market filter: HSI above 200‑day MA
- RS calculation vs HSI, volume filter, chunked Telegram messages
- Scheduled daily at 08:15 HKT (Mon–Fri)
"""

import io
import os
import html
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import certifi
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ------------------------------------------------------------------ Config
@dataclass
class Config:
    # Data & filtering
    history_days: int = 420
    rs_ma_days: int = 200               # for market filter
    min_volume: int = 500_000           # 50‑day avg volume (HKD stocks usually higher)
    rs_percentile: float = 70.0

    # VCP parameters
    lookback_contractions: int = 160
    order_strict: int = 5
    order_practical: int = 4
    min_pct: float = 0.03
    max_pct_strict: float = 0.30
    max_pct_practical: float = 0.35
    dryup_ratio_strict: float = 0.70
    dryup_ratio_practical: float = 0.80
    near_pivot_pct: float = 5.0
    breakout_vol_mult: float = 1.4

    # Benchmarks (yahoo tickers for Hong Kong indices)
    benchmark_ticker: str = '^HSI'      # Hang Seng Index
    # Universe: we will auto-fetch constituents from Wikipedia / csv fallback

    # Output
    output_dir: str = 'output'

    # Telegram
    bot_token: str = field(default_factory=lambda: os.getenv('TELEGRAM_BOT_TOKEN', ''))
    chat_id: str = field(default_factory=lambda: os.getenv('TELEGRAM_CHAT_ID', ''))

config = Config()

# ------------------------------------------------------------------ Helpers
def script_dir():
    return os.path.dirname(os.path.abspath(__file__))

def output_dir():
    out = os.path.join(script_dir(), config.output_dir)
    os.makedirs(out, exist_ok=True)
    return out

HEADERS = {'User-Agent': 'Mozilla/5.0'}

import requests
import pandas as pd
from io import StringIO

import requests
import pandas as pd
from io import StringIO

def get_hk_index_tickers():
    """
    智能获取恒生指数、国企指数、恒生科技指数的全部成分股。
    优先从 Yahoo Finance 抓取，失败时使用 GitHub 备份，最后使用本地缓存。
    所有代码自动补足 .HK 后缀，直接兼容 yfinance。
    """

    def fetch_hsi_from_yahoo():
        """从 Yahoo Finance 抓取恒生指数成分股 (^HSI)"""
        try:
            url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=HSI_components_latest&count=100"
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            rows = data['finance']['result'][0]['quotes']
            codes = [f"{str(row['symbol']).replace('.HK', '').zfill(4)}.HK" for row in rows]
            return codes if codes else None
        except Exception as e:
            print(f'Yahoo Finance HSI 抓取失败: {e}')
            return None

    def fetch_hsi_from_github():
        """从 GitHub 备份抓取恒生指数成分股"""
        try:
            url = "https://yfiua.github.io/index-constituents/constituents-hsi.csv"
            df = pd.read_csv(url)
            if 'Symbol' in df.columns:
                codes = [f"{str(s).strip().replace('.HK', '').zfill(4)}.HK" for s in df['Symbol'].dropna()]
                return codes
        except Exception as e:
            print(f'GitHub HSI 备份抓取失败: {e}')
        return None

    def fetch_hscei_from_yahoo():
        """抓取恒生中国企业指数成分股"""
        try:
            url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=HSCEI_components_latest&count=100"
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            rows = data['finance']['result'][0]['quotes']
            codes = [f"{str(row['symbol']).replace('.HK', '').zfill(4)}.HK" for row in rows]
            return codes if codes else None
        except Exception as e:
            print(f'Yahoo Finance HSCEI 抓取失败: {e}')
            return None

    def fetch_hstech_from_yahoo():
        """抓取恒生科技指数成分股"""
        try:
            url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=HSTECH_components_latest&count=100"
            headers = {'User-Agent': 'Mozilla/5.0'}
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            rows = data['finance']['result'][0]['quotes']
            codes = [f"{str(row['symbol']).replace('.HK', '').zfill(4)}.HK" for row in rows]
            return codes if codes else None
        except Exception as e:
            print(f'Yahoo Finance HSTECH 抓取失败: {e}')
            return None

    # 最终本地缓存 (2025年最新)
    FALLBACK_HSI = [
        '0005.HK','0011.HK','0016.HK','0027.HK','0066.HK','0083.HK','0101.HK',
        '0175.HK','0241.HK','0267.HK','0288.HK','0291.HK','0316.HK','0386.HK',
        '0388.HK','0669.HK','0700.HK','0762.HK','0823.HK','0857.HK','0883.HK',
        '0939.HK','0941.HK','0960.HK','0968.HK','0981.HK','0992.HK','1038.HK',
        '1044.HK','1088.HK','1093.HK','1109.HK','1113.HK','1177.HK','1211.HK',
        '1299.HK','1378.HK','1398.HK','1658.HK','1801.HK','1876.HK','1919.HK',
        '1928.HK','1997.HK','2007.HK','2015.HK','2020.HK','2269.HK','2313.HK',
        '2318.HK','2319.HK','2331.HK','2382.HK','2388.HK','2628.HK','2899.HK',
        '3690.HK','3968.HK','3988.HK','6098.HK','6618.HK','6690.HK','6862.HK',
        '9618.HK','9626.HK','9888.HK','9961.HK','9988.HK'
    ]
    
    FALLBACK_HSTECH = [
        '0700.HK','9988.HK','3690.HK','1810.HK','1211.HK','9999.HK','9618.HK',
        '2015.HK','1024.HK','6618.HK','2382.HK','1797.HK','2013.HK','9698.HK',
        '9868.HK','9866.HK','6186.HK','0268.HK','0388.HK','0241.HK','0020.HK',
        '1347.HK','0981.HK','0772.HK','1833.HK','6060.HK','0285.HK','9923.HK',
        '9961.HK','9626.HK','2518.HK'
    ]

    FALLBACK_HSCEI = [
        '0939.HK','1398.HK','3988.HK','1288.HK','9988.HK','3690.HK','0941.HK',
        '0857.HK','0883.HK','0386.HK','0728.HK','0762.HK','2318.HK','2628.HK',
        '2601.HK','1336.HK','2328.HK','0998.HK','3968.HK','1658.HK','1810.HK',
        '1211.HK','0700.HK','9618.HK','9999.HK','2020.HK','2331.HK','2313.HK',
        '2688.HK','0291.HK','1109.HK','3328.HK','6030.HK','6837.HK','2333.HK',
        '1772.HK','1776.HK','3908.HK','6886.HK','2611.HK','6066.HK','6881.HK',
        '6098.HK','6618.HK','6690.HK','6862.HK','9626.HK'
    ]

    all_codes = set()

    # 1. 恒生指数
    print('正在抓取 HSI 成分股...')
    hsi_codes = fetch_hsi_from_yahoo() or fetch_hsi_from_github()
    if hsi_codes:
        all_codes.update(hsi_codes)
        print(f'  获取到 {len(hsi_codes)} 只成分股')
    else:
        print(f'  HSI 所有在线源失效，使用本地缓存 ({len(FALLBACK_HSI)} 只)')
        all_codes.update(FALLBACK_HSI)

    # 2. 恒生中国企业指数
    print('正在抓取 HSCEI 成分股...')
    hscei_codes = fetch_hscei_from_yahoo()
    if hscei_codes:
        all_codes.update(hscei_codes)
        print(f'  获取到 {len(hscei_codes)} 只成分股')
    else:
        print(f'  HSCEI 抓取失败，使用本地缓存 ({len(FALLBACK_HSCEI)} 只)')
        all_codes.update(FALLBACK_HSCEI)

    # 3. 恒生科技指数
    print('正在抓取 HSTECH 成分股...')
    hstech_codes = fetch_hstech_from_yahoo()
    if hstech_codes:
        all_codes.update(hstech_codes)
        print(f'  获取到 {len(hstech_codes)} 只成分股')
    else:
        print(f'  HSTECH 抓取失败，使用本地缓存 ({len(FALLBACK_HSTECH)} 只)')
        all_codes.update(FALLBACK_HSTECH)

    return sorted(list(all_codes))
def get_price_df(symbol, start=None, end=None):
    """Download and clean OHLCV dataframe."""
    if start is None:
        start = (datetime.now() - timedelta(days=config.history_days)).strftime('%Y-%m-%d')
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')

    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False, threads=False)
    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        # Try to extract the ticker-specific data if possible
        if symbol in df.columns.get_level_values(1):
            df = df.xs(symbol, axis=1, level=1)
        else:
            # fallback: use first level (might be OHLCV without ticker)
            df.columns = df.columns.get_level_values(0)

    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required:
        if col not in df.columns:
            return None

    if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']

    return df.dropna().copy()


def market_ok(df_bench):
    """Check if benchmark (HSI) is above its 200‑day MA."""
    if df_bench is None or len(df_bench) < config.rs_ma_days:
        return False
    close = df_bench['Adj Close']
    ma200 = close.rolling(window=config.rs_ma_days).mean().iloc[-1]
    return pd.notna(ma200) and close.iloc[-1] > ma200


# -------------------------------- RS & Trend (same as before, adjusted for RS)
def compute_rs(symbol, bench_df):
    """RS = (stock return / benchmark return) over common period."""
    df = get_price_df(symbol)
    if df is None or len(df) < 50:
        return None, None

    common = df.index.intersection(bench_df.index)
    if len(common) < 50:
        return None, None

    stock_close = df.loc[common, 'Adj Close']
    bench_close = bench_df.loc[common, 'Adj Close']
    stock_ret = stock_close.iloc[-1] / stock_close.iloc[0]
    bench_ret = bench_close.iloc[-1] / bench_close.iloc[0]
    rs = stock_ret / bench_ret
    return rs, df


def local_extrema(series, order=4):
    highs, lows = [], []
    vals = series.values
    idx = series.index
    for i in range(order, len(series) - order):
        window = vals[i - order:i + order + 1]
        center = vals[i]
        if np.isfinite(center) and center == np.max(window):
            highs.append((idx[i], float(center), i))
        if np.isfinite(center) and center == np.min(window):
            lows.append((idx[i], float(center), i))
    return highs, lows


def extract_contractions(df, order, max_lookback, min_pct, max_pct):
    recent = df.tail(max_lookback).copy()
    highs, lows = local_extrema(recent['Adj Close'], order=order)
    contractions = []
    for h_date, h_price, h_pos in highs:
        next_lows = [(d, p, pos) for d, p, pos in lows if pos > h_pos]
        if not next_lows:
            continue
        l_date, l_price, l_pos = next_lows[0]
        days = l_pos - h_pos
        if days < 3:
            continue
        pct = (h_price - l_price) / h_price
        if min_pct <= pct <= max_pct:
            avg_vol = recent['Volume'].iloc[h_pos:l_pos + 1].mean()
            contractions.append({
                'high_date': h_date,
                'low_date': l_date,
                'high_pos': h_pos,
                'low_pos': l_pos,
                'high_price': h_price,
                'low_price': l_price,
                'contraction_pct': pct,
                'avg_pullback_volume': avg_vol,
            })
    return contractions[-4:], recent


def trend_template(df, strict=False):
    for x in [20, 50, 150, 200]:
        df[f'SMA_{x}'] = df['Adj Close'].rolling(window=x).mean()

    close = float(df['Adj Close'].iloc[-1])
    ma20 = float(df['SMA_20'].iloc[-1])
    ma50 = float(df['SMA_50'].iloc[-1])
    ma150 = float(df['SMA_150'].iloc[-1])
    ma200 = float(df['SMA_200'].iloc[-1])
    ma200_20 = float(df['SMA_200'].iloc[-20]) if len(df) >= 220 and pd.notna(df['SMA_200'].iloc[-20]) else np.nan
    high_52w = float(df['High'].tail(250).max())
    low_52w = float(df['Low'].tail(250).min())

    if strict:
        ok = all([
            pd.notna(ma20), pd.notna(ma50), pd.notna(ma150), pd.notna(ma200),
            close > ma50 > ma150 > ma200,
            ma150 > ma200,
            pd.notna(ma200_20) and ma200 > ma200_20,
            close >= 0.80 * high_52w,
            close >= 1.30 * low_52w,
            close > 1.0,   # HK stocks often above 1 HKD
        ])
    else:
        ok = all([
            pd.notna(ma50), pd.notna(ma150), pd.notna(ma200),
            close > ma50 > ma150 > ma200,
            ma150 > ma200,
            pd.notna(ma200_20) and ma200 >= ma200_20,
            close >= 0.75 * high_52w,
            close >= 1.25 * low_52w,
            close > 1.0,
        ])
    return {
        'ok': ok,
        'close': close, 'ma20': ma20, 'ma50': ma50, 'ma150': ma150, 'ma200': ma200,
        'high_52w': high_52w, 'low_52w': low_52w,
    }


def evaluate_strict_vcp(df):
    if df is None or len(df) < 250:
        return None
    t = trend_template(df, strict=True)
    if not t['ok']:
        return None
    contractions, recent = extract_contractions(
        df, order=config.order_strict, max_lookback=config.lookback_contractions,
        min_pct=config.min_pct, max_pct=config.max_pct_strict
    )
    if len(contractions) < 2:
        return None
    contraction_pcts = [c['contraction_pct'] for c in contractions]
    if not all(contraction_pcts[i] > contraction_pcts[i + 1] for i in range(len(contraction_pcts) - 1)):
        return None
    volume_seq = [c['avg_pullback_volume'] for c in contractions]
    avg50_volume = float(recent['Volume'].tail(50).mean())
    if avg50_volume < config.min_volume:
        return None
    last_vol_ratio = volume_seq[-1] / avg50_volume
    if last_vol_ratio > config.dryup_ratio_strict:
        return None
    decreasing_pairs = sum(1 for i in range(len(volume_seq)-1) if volume_seq[i] >= volume_seq[i+1])
    if decreasing_pairs < max(1, len(volume_seq) - 2):
        return None

    last = contractions[-1]
    pivot = float(recent['High'].iloc[last['low_pos']:].max())
    latest_close = float(recent['Adj Close'].iloc[-1])
    latest_volume = float(recent['Volume'].iloc[-1])
    dist_pct = ((pivot - latest_close) / pivot) * 100 if pivot > 0 else np.nan
    near_pivot = latest_close >= pivot * 0.97
    breakout = (latest_close > pivot) and (latest_volume >= config.breakout_vol_mult * avg50_volume)

    setup_type = 'Strict VCP'
    if breakout:
        setup_type = 'Strict Breakout'
    elif near_pivot:
        setup_type = 'Strict Near Pivot'

    return {
        'Mode': 'Strict', 'Setup Type': setup_type,
        'Close': round(t['close'], 2), 'Pivot': round(pivot, 2),
        'Distance to Pivot %': round(dist_pct, 2) if pd.notna(dist_pct) else np.nan,
        'Near Pivot': bool(near_pivot), 'Breakout Now': bool(breakout),
        'Contractions': ' | '.join(f'{x*100:.1f}%' for x in contraction_pcts),
        'Volume Dry-Up Ratio': round(last_vol_ratio, 2),
        'Avg Pullback Volumes': ' | '.join(f'{int(v):,}' for v in volume_seq),
        '52W High': round(t['high_52w'], 2), '52W Low': round(t['low_52w'], 2),
        '50MA': round(t['ma50'], 2), '150MA': round(t['ma150'], 2), '200MA': round(t['ma200'], 2),
    }


def evaluate_practical_vcp(df):
    if df is None or len(df) < 250:
        return None
    t = trend_template(df, strict=False)
    if not t['ok']:
        return None
    contractions, recent = extract_contractions(
        df, order=config.order_practical, max_lookback=config.lookback_contractions,
        min_pct=config.min_pct, max_pct=config.max_pct_practical
    )
    if len(contractions) < 2:
        return None
    contraction_pcts = [c['contraction_pct'] for c in contractions]
    first_last_tighter = contraction_pcts[-1] < contraction_pcts[0]
    non_expanding = sum(contraction_pcts[i+1] <= contraction_pcts[i] for i in range(len(contraction_pcts)-1))
    if not (first_last_tighter and non_expanding >= max(1, len(contraction_pcts)-2)):
        return None
    volume_seq = [c['avg_pullback_volume'] for c in contractions]
    avg50_volume = float(recent['Volume'].tail(50).mean())
    if avg50_volume < config.min_volume:
        return None
    dryup_ratio = volume_seq[-1] / avg50_volume
    if dryup_ratio > config.dryup_ratio_practical:
        return None

    last = contractions[-1]
    pivot = float(recent['High'].iloc[last['low_pos']:].max())
    latest_close = float(recent['Adj Close'].iloc[-1])
    latest_volume = float(recent['Volume'].iloc[-1])
    dist_pct = ((pivot - latest_close) / pivot) * 100 if pivot > 0 else np.nan
    near_pivot = pd.notna(dist_pct) and 0 <= dist_pct <= config.near_pivot_pct
    breakout = latest_close > pivot and latest_volume >= config.breakout_vol_mult * avg50_volume

    setup_type = 'Watchlist'
    if breakout:
        setup_type = 'Breakout Today'
    elif near_pivot:
        setup_type = 'Near Pivot'

    return {
        'Mode': 'Practical', 'Setup Type': setup_type,
        'Close': round(t['close'], 2), 'Pivot': round(pivot, 2),
        'Distance to Pivot %': round(dist_pct, 2) if pd.notna(dist_pct) else np.nan,
        'Near Pivot': bool(near_pivot), 'Breakout Now': bool(breakout),
        'Contractions': ' | '.join(f'{x*100:.1f}%' for x in contraction_pcts),
        'Volume Dry-Up Ratio': round(dryup_ratio, 2),
        'Avg Pullback Volumes': ' | '.join(f'{int(v):,}' for v in volume_seq),
        '52W High': round(t['high_52w'], 2), '52W Low': round(t['low_52w'], 2),
        '50MA': round(t['ma50'], 2), '150MA': round(t['ma150'], 2), '200MA': round(t['ma200'], 2),
    }


# ------------------------------ Telegram helpers
def split_long_message(text: str, max_len=4000) -> List[str]:
    if len(text) <= max_len:
        return [text]
    chunks = []
    lines = text.split('\n')
    current = ''
    for line in lines:
        if len(current) + len(line) + 1 > max_len:
            chunks.append(current)
            current = line
        else:
            current += '\n' + line if current else line
    if current:
        chunks.append(current)
    return chunks


def format_telegram_message(strict_df, practical_df, combined_df):
    if combined_df.empty:
        return '<b>港股 VCP 篩選</b>\n今日無符合條件個股。'

    lines = ['<b>港股 VCP 篩選</b>']
    lines.append(f"<b>合計:</b> {len(combined_df)} | <b>Strict:</b> {len(strict_df)} | <b>Practical:</b> {len(practical_df)}")

    sections = [
        ('🔥 Strict Breakout', strict_df[strict_df['Setup Type'] == 'Strict Breakout'].head(5)),
        ('🎯 Strict Near Pivot', strict_df[strict_df['Setup Type'].isin(['Strict Near Pivot','Strict VCP'])].head(5)),
        ('🚀 Practical Breakout', practical_df[practical_df['Setup Type'] == 'Breakout Today'].head(5)),
        ('👀 Practical Near Pivot', practical_df[practical_df['Setup Type'] == 'Near Pivot'].head(5)),
        ('📌 Practical Watchlist', practical_df[practical_df['Setup Type'] == 'Watchlist'].head(5)),
    ]

    for title, subset in sections:
        if subset.empty:
            continue
        lines.append(f"\n<b>{html.escape(title)}</b>")
        for _, row in subset.iterrows():
            stock = html.escape(str(row['Stock']))
            contractions = html.escape(str(row['Contractions']))
            dist = row['Distance to Pivot %']
            dry = row['Volume Dry-Up Ratio']
            lines.append(
                f"• <b>{stock}</b> | RS {row['RS_Rating']:.0f} | Pivot {row['Pivot']:.2f} | "
                f"距離樞紐 {dist:.2f}% | 量縮比 {dry:.2f} | {contractions}"
            )
    return '\n'.join(lines)


def send_telegram_messages(messages: List[str]):
    for msg in messages:
        if not config.bot_token or not config.chat_id:
            print('Telegram credentials not set, skip.')
            return
        url = f'https://api.telegram.org/bot{config.bot_token}/sendMessage'
        payload = {
            'chat_id': config.chat_id,
            'text': msg,
            'parse_mode': 'HTML',
            'disable_web_page_preview': True,
        }
        r = requests.post(url, data=payload, timeout=30)
        r.raise_for_status()
        print(f'Sent chunk len={len(msg)}')


# --------------------------------- Main
def main():
    os.makedirs(output_dir(), exist_ok=True)

    # 1. Get Hang Seng Index data for market filter & RS benchmark
    print('Downloading benchmark...')
    bench_df = get_price_df(config.benchmark_ticker)
    if bench_df is None or bench_df.empty:
        raise ValueError('Cannot download HSI data.')

    if not market_ok(bench_df):
        msg = '<b>港股 VCP 篩選</b>\n❌ 大市過濾：恒指低於 200 日均線，今日不發訊號。'
        send_telegram_messages([msg])
        print('Market filter blocked.')
        return

    # 2. Fetch stock universe
    print('Fetching index constituents...')
    tickers = get_hk_index_tickers()
    print(f'Total tickers: {len(tickers)}')

    # 3. Compute RS and cache data
    rs_rows = []
    cache = {}
    for tkr in tickers:
        try:
            rs, df = compute_rs(tkr, bench_df)
            if rs is None or df is None:
                continue
            avg_vol = df['Volume'].rolling(50).mean().iloc[-1]
            if pd.isna(avg_vol) or avg_vol < config.min_volume:
                continue
            rs_rows.append({'Ticker': tkr, 'RS_Multiple': rs})
            cache[tkr] = df
            print(f'{tkr}: RS={rs:.2f}')
        except Exception as e:
            print(f'Error {tkr}: {e}')

    if not rs_rows:
        print('No stocks passed RS & volume filters.')
        return

    rs_df = pd.DataFrame(rs_rows)
    rs_df['RS_Rating'] = rs_df['RS_Multiple'].rank(pct=True) * 100
    rs_df = rs_df[rs_df['RS_Rating'] >= config.rs_percentile].copy()
    rs_df = rs_df.sort_values('RS_Rating', ascending=False)

    # 4. VCP evaluation
    strict_rows, practical_rows = [], []
    for stock in rs_df['Ticker']:
        df = cache.get(stock)
        rs_val = rs_df.loc[rs_df['Ticker'] == stock, 'RS_Rating'].iloc[0]
        s = evaluate_strict_vcp(df)
        if s:
            strict_rows.append({'Stock': stock, 'RS_Rating': round(rs_val, 2), **s})
        p = evaluate_practical_vcp(df)
        if p:
            practical_rows.append({'Stock': stock, 'RS_Rating': round(rs_val, 2), **p})

    all_cols = ['Stock','RS_Rating','Mode','Setup Type','Close','Pivot',
                'Distance to Pivot %','Near Pivot','Breakout Now','Contractions',
                'Volume Dry-Up Ratio','Avg Pullback Volumes','52W High','52W Low',
                '50MA','150MA','200MA']

    strict_df = pd.DataFrame(strict_rows, columns=all_cols) if strict_rows else pd.DataFrame(columns=all_cols)
    practical_df = pd.DataFrame(practical_rows, columns=all_cols) if practical_rows else pd.DataFrame(columns=all_cols)

    combined_df = pd.concat([strict_df, practical_df], ignore_index=True)
    if not combined_df.empty:
        priority = {
            'Strict Breakout':0,'Strict Near Pivot':1,'Strict VCP':2,
            'Breakout Today':3,'Near Pivot':4,'Watchlist':5,
        }
        combined_df['Rank'] = combined_df['Setup Type'].map(priority).fillna(9)
        combined_df = combined_df.sort_values(['Rank','RS_Rating','Distance to Pivot %'],
                                              ascending=[True,False,True]).drop(columns=['Rank'])
        strict_df = strict_df.sort_values(['RS_Rating','Distance to Pivot %'], ascending=[False,True])
        practical_df = practical_df.sort_values(['RS_Rating','Distance to Pivot %'], ascending=[False,True])

    # 5. Telegram
    msg = format_telegram_message(strict_df, practical_df, combined_df)
    chunks = split_long_message(msg)
    try:
        send_telegram_messages(chunks)
    except Exception as e:
        print(f'Telegram send failed: {e}')

    # 6. Save output (optional)
    combined_df.to_csv(os.path.join(output_dir(), 'hk_vcp_combined.csv'), index=False)
    print('Done.')


if __name__ == '__main__':
    main()
