# -*- coding: utf-8 -*-
"""
Market Regime Dashboard (Public Version)
DB에서 데이터 조회만 하는 공개용 앱

핵심 로직 없음 - 순수 시각화만 담당
"""
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pymysql
import plotly.graph_objects as go
import os

# =============================================================================
# DB Connection
# =============================================================================
def get_db_connection():
    """DB 연결 (환경변수에서 읽기)"""
    return pymysql.connect(
        host=os.getenv('MARIADB_HOST', 'gdpresent.synology.me'),
        port=int(os.getenv('MARIADB_PORT', 3306)),
        user=os.getenv('MARIADB_USER', 'regime_user'),
        password=os.getenv('MARIADB_PASSWORD', ''),
        database=os.getenv('MARIADB_DATABASE', 'regime_db'),
        charset='utf8mb4'
    )


def query_df(sql: str) -> pd.DataFrame:
    """SQL 실행 후 DataFrame 반환"""
    conn = get_db_connection()
    df = pd.read_sql(sql, conn)
    conn.close()
    return df


# =============================================================================
# Data Loading Functions
# =============================================================================
@st.cache_data(ttl=3600)
def load_regime_summary():
    """국가별 regime 현황"""
    sql = """
    SELECT country, first_regime, fresh_regime, smart_regime, 
           cli_level, cli_momentum, data_month, date
    FROM regime_summary
    WHERE date = (SELECT MAX(date) FROM regime_summary)
    ORDER BY country
    """
    return query_df(sql)


@st.cache_data(ttl=3600)
def load_portfolio_weights():
    """현재 포트폴리오 비중"""
    sql = """
    SELECT country, ticker, weight, regime, score, date
    FROM portfolio_weights
    WHERE date = (SELECT MAX(date) FROM portfolio_weights)
    AND weight > 0.001
    ORDER BY weight DESC
    """
    return query_df(sql)


@st.cache_data(ttl=3600)
def load_backtest_metrics():
    """백테스트 성과 지표"""
    sql = """
    SELECT * FROM backtest_metrics
    WHERE date = (SELECT MAX(date) FROM backtest_metrics)
    """
    return query_df(sql)


@st.cache_data(ttl=3600)
def load_cumulative_returns():
    """누적수익률 시계열"""
    sql = """
    SELECT date, strategy_return, benchmark_return
    FROM cumulative_returns
    ORDER BY date
    """
    return query_df(sql)


@st.cache_data(ttl=3600)
def load_market_indicators():
    """Market Indicators"""
    sql = """
    SELECT * FROM market_indicators
    WHERE date = (SELECT MAX(date) FROM market_indicators)
    """
    return query_df(sql)


@st.cache_data(ttl=3600)
def load_global_index_returns():
    """글로벌 지수 수익률"""
    sql = """
    SELECT index_name, ticker, return_1d, return_1w, return_1m, 
           return_3m, return_ytd, return_1y, last_price, date
    FROM global_index_returns
    WHERE date = (SELECT MAX(date) FROM global_index_returns)
    ORDER BY index_name
    """
    return query_df(sql)


@st.cache_data(ttl=3600)
def load_sector_returns():
    """섹터별 수익률"""
    sql = """
    SELECT sector, ticker, return_1d, return_1w, return_1m, 
           return_3m, return_ytd, return_1y, date
    FROM sector_returns
    WHERE date = (SELECT MAX(date) FROM sector_returns)
    ORDER BY sector
    """
    return query_df(sql)


@st.cache_data(ttl=3600)
def load_rebalancing_history():
    """리밸런싱 이력"""
    sql = """
    SELECT trade_date, country, weight
    FROM rebalancing_history
    ORDER BY trade_date DESC, weight DESC
    LIMIT 100
    """
    return query_df(sql)


# =============================================================================
# Page Config
# =============================================================================
st.set_page_config(
    page_title="Market Regime Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Custom CSS
# =============================================================================
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .regime-expansion { background-color: #2ca02c; color: white; padding: 5px 10px; border-radius: 5px; }
    .regime-recovery { background-color: #ffce30; color: black; padding: 5px 10px; border-radius: 5px; }
    .regime-slowdown { background-color: #ff7f0e; color: white; padding: 5px 10px; border-radius: 5px; }
    .regime-contraction { background-color: #d62728; color: white; padding: 5px 10px; border-radius: 5px; }
    @media (max-width: 768px) {
        .main-header { font-size: 1.5rem; }
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Main Content
# =============================================================================
st.markdown('<div class="main-header">📊 Market Regime Dashboard</div>', unsafe_allow_html=True)

# 데이터 기준일
try:
    regime_df = load_regime_summary()
    if not regime_df.empty:
        data_date = regime_df['date'].iloc[0]
        st.caption(f"📅 데이터 기준일: {data_date}")
    else:
        st.warning("데이터가 없습니다. 파이프라인 실행이 필요합니다.")
        st.stop()
except Exception as e:
    st.error(f"DB 연결 실패: {e}")
    st.info("환경변수 설정을 확인해주세요: MARIADB_HOST, MARIADB_PASSWORD 등")
    st.stop()

st.markdown("---")

# =============================================================================
# Global Index Returns
# =============================================================================
st.subheader("📈 글로벌 지수 현황")

idx_tab1, idx_tab2 = st.tabs(["🌐 글로벌 지수", "🏭 섹터별 현황"])

with idx_tab1:
    index_df = load_global_index_returns()
    if not index_df.empty:
        display_df = index_df[['index_name', 'return_1d', 'return_1w', 'return_1m', 'return_ytd', 'return_1y', 'last_price']].copy()
        display_df.columns = ['지수', '1D', '1W', '1M', 'YTD', '1Y', '종가']
        
        # 수익률 포맷팅
        for col in ['1D', '1W', '1M', 'YTD', '1Y']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
        display_df['종가'] = display_df['종가'].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "-")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("지수 데이터 없음")

with idx_tab2:
    sector_df = load_sector_returns()
    if not sector_df.empty:
        display_df = sector_df[['sector', 'return_1d', 'return_1w', 'return_1m', 'return_ytd']].copy()
        display_df.columns = ['섹터', '1D', '1W', '1M', 'YTD']
        
        for col in ['1D', '1W', '1M', 'YTD']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("섹터 데이터 없음")

st.markdown("---")

# =============================================================================
# Regime Summary
# =============================================================================
st.subheader("🌍 현재 국면 요약")

def color_regime(val):
    colors = {
        '팽창': 'background-color: #2ca02c; color: white',
        '회복': 'background-color: #ffce30; color: black',
        '둔화': 'background-color: #ff7f0e; color: white',
        '침체': 'background-color: #d62728; color: white',
        'Cash': 'background-color: #ffb347; color: black',
        'Half': 'background-color: #9467bd; color: white',
        'Skipped': 'background-color: #f0f0f0; color: black'
    }
    return colors.get(val, '')

regime_display = regime_df[['country', 'first_regime', 'fresh_regime', 'smart_regime']].copy()
regime_display.columns = ['국가', 'First', 'Fresh', 'Smart']

styled_df = regime_display.style.applymap(color_regime, subset=['First', 'Fresh', 'Smart'])
st.dataframe(styled_df, use_container_width=True, hide_index=True)

st.markdown("---")

# =============================================================================
# ETF Rotation Strategy
# =============================================================================
st.subheader("🎯 ETF Rotation Strategy")
st.caption("📊 v4: BiInvVol | Top 2 집중 | 변동성 낮은 국가 우선")

st.markdown("#### 📍 현재 포지션")

portfolio_df = load_portfolio_weights()
if not portfolio_df.empty:
    # CASH 제외한 투자 포지션
    investing = portfolio_df[portfolio_df['country'] != 'CASH']
    cash_row = portfolio_df[portfolio_df['country'] == 'CASH']
    
    if not investing.empty:
        pos_display = investing[['country', 'ticker', 'regime', 'score', 'weight']].copy()
        pos_display.columns = ['국가', 'Ticker', 'Regime', 'Score', '비중']
        pos_display['비중'] = pos_display['비중'].apply(lambda x: f"{x:.1%}")
        
        styled_pos = pos_display.style.applymap(color_regime, subset=['Regime'])
        
        pos_cols = st.columns([3, 1])
        with pos_cols[0]:
            st.dataframe(styled_pos, hide_index=True, use_container_width=True)
        with pos_cols[1]:
            cash_pct = cash_row['weight'].iloc[0] if not cash_row.empty else 0
            st.metric("현금 비중", f"{cash_pct:.1%}")
            st.caption(f"마지막 업데이트: {portfolio_df['date'].iloc[0]}")
    else:
        st.warning("추천 포지션 없음 (전액 CASH)")
    
    # Pie Chart
    with st.expander("🥧 포트폴리오 구성"):
        pie_data = portfolio_df[portfolio_df['weight'] > 0.001]
        if not pie_data.empty:
            fig_pie = go.Figure(data=[go.Pie(
                labels=pie_data['country'].tolist(),
                values=pie_data['weight'].tolist(),
                hole=0.4,
                textinfo='label+percent',
                hovertemplate='%{label}: %{percent}<extra></extra>'
            )])
            fig_pie.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_pie, use_container_width=True)
else:
    st.info("포트폴리오 데이터 없음")

# =============================================================================
# Cumulative Returns Chart
# =============================================================================
st.markdown("#### 📈 누적 수익률 (Backtest)")

cum_df = load_cumulative_returns()
if not cum_df.empty:
    fig_cum = go.Figure()
    
    fig_cum.add_trace(go.Scatter(
        x=cum_df['date'], y=cum_df['strategy_return'],
        name='Strategy (FIRST)',
        line=dict(color='#2ca02c', width=2),
        hovertemplate='%{x|%Y-%m-%d}<br>Return: %{y:.1%}<extra></extra>'
    ))
    
    fig_cum.add_trace(go.Scatter(
        x=cum_df['date'], y=cum_df['benchmark_return'],
        name='ACWI (BM)',
        line=dict(color='silver', width=2, dash='dash'),
        hovertemplate='%{x|%Y-%m-%d}<br>Return: %{y:.1%}<extra></extra>'
    ))
    
    fig_cum.update_layout(
        height=350,
        xaxis_title='Date',
        yaxis_title='Cumulative Return',
        yaxis_tickformat='.0%',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified',
        margin=dict(t=30, b=30, l=50, r=20)
    )
    
    st.plotly_chart(fig_cum, use_container_width=True)
    
    # 성과 지표
    metrics_df = load_backtest_metrics()
    if not metrics_df.empty:
        m = metrics_df.iloc[0]
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        with perf_col1:
            delta = (m['cagr'] - m['bm_cagr']) * 100 if pd.notna(m['bm_cagr']) else 0
            st.metric("CAGR", f"{m['cagr']:.1%}", delta=f"{delta:.1f}%p vs BM")
        with perf_col2:
            delta = m['sharpe'] - m['bm_sharpe'] if pd.notna(m['bm_sharpe']) else 0
            st.metric("Sharpe", f"{m['sharpe']:.2f}", delta=f"{delta:+.2f} vs BM")
        with perf_col3:
            delta = (m['mdd'] - m['bm_mdd']) * 100 if pd.notna(m['bm_mdd']) else 0
            st.metric("MDD", f"{m['mdd']:.1%}", delta=f"{delta:+.1f}%p")
        with perf_col4:
            st.metric("Vol", f"{m['volatility']:.1%}")
else:
    st.info("누적수익률 데이터 없음")

st.markdown("---")

# =============================================================================
# Rebalancing History
# =============================================================================
with st.expander("📅 리밸런싱 이력 (최근)"):
    rebal_df = load_rebalancing_history()
    if not rebal_df.empty:
        # Pivot to wide format
        pivot_df = rebal_df.pivot(index='trade_date', columns='country', values='weight')
        pivot_df = (pivot_df * 100).round(1)
        pivot_df = pivot_df.tail(10)
        st.dataframe(pivot_df, use_container_width=True)
    else:
        st.info("리밸런싱 이력 없음")

# =============================================================================
# Market Indicators (Sidebar)
# =============================================================================
st.sidebar.title("📊 Market Indicators")

indicators_df = load_market_indicators()
if not indicators_df.empty:
    m = indicators_df.iloc[0]
    
    if pd.notna(m['fear_greed']):
        fg_color = '#2ca02c' if m['fear_greed'] > 50 else '#d62728'
        st.sidebar.metric("Fear & Greed", f"{m['fear_greed']}", m['fear_greed_text'])
    
    if pd.notna(m['vix']):
        st.sidebar.metric("VIX", f"{m['vix']:.2f}")
    
    if pd.notna(m['yield_spread']):
        st.sidebar.metric("Yield Spread (10Y-2Y)", f"{m['yield_spread']:.2f}%")
    
    if pd.notna(m['dxy']):
        st.sidebar.metric("DXY", f"{m['dxy']:.2f}")
    
    if pd.notna(m['breadth_position']):
        st.sidebar.metric("Market Breadth", f"{m['breadth_position']:.1%}", m['breadth_status'])

st.sidebar.markdown("---")
st.sidebar.caption("Powered by MariaDB on Synology NAS")
