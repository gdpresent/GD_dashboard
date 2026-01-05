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
import plotly.express as px
import yfinance as yf
import os

# =============================================================================
# Constants
# =============================================================================
COUNTRY_MAP = {
    'USA': {'name': '미국', 'ticker': 'SPY'},
    'Korea': {'name': '한국', 'ticker': 'EWY'},
    'Japan': {'name': '일본', 'ticker': 'EWJ'},
    'China': {'name': '중국', 'ticker': 'FXI'},
    'Germany': {'name': '독일', 'ticker': 'EWG'},
    'France': {'name': '프랑스', 'ticker': 'EWQ'},
    'UK': {'name': '영국', 'ticker': 'EWU'},
    'India': {'name': '인도', 'ticker': 'INDA'},
    'Brazil': {'name': '브라질', 'ticker': 'EWZ'}
}

REGIME_COLORS = {
    '팽창': '#2ca02c', '회복': '#ffce30', '둔화': '#ff7f0e', '침체': '#d62728',
    'Cash': '#ffb347', 'Half': '#9467bd', 'Skipped': '#f0f0f0'
}

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
    sql = """SELECT * FROM backtest_metrics WHERE date = (SELECT MAX(date) FROM backtest_metrics)"""
    return query_df(sql)

@st.cache_data(ttl=3600)
def load_cumulative_returns():
    sql = """SELECT date, strategy_return, benchmark_return FROM cumulative_returns ORDER BY date"""
    return query_df(sql)

@st.cache_data(ttl=3600)
def load_market_indicators():
    sql = """SELECT * FROM market_indicators WHERE date = (SELECT MAX(date) FROM market_indicators)"""
    return query_df(sql)

@st.cache_data(ttl=3600)
def load_global_index_returns():
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
    sql = """
    SELECT trade_date, country, weight
    FROM rebalancing_history
    ORDER BY trade_date DESC
    LIMIT 200
    """
    return query_df(sql)

@st.cache_data(ttl=3600)
def load_country_regimes(country: str):
    """국가별 국면 시계열 (Business Cycle Clock용)"""
    sql = f"""
    SELECT trade_date, data_month, cli_level, cli_momentum, 
           level_first, momentum_first, exp1_regime, exp2_regime, exp3_regime
    FROM country_regimes
    WHERE date = (SELECT MAX(date) FROM country_regimes)
    AND country = '{country}'
    ORDER BY trade_date
    """
    return query_df(sql)

@st.cache_data(ttl=3600)
def load_country_cumulative_returns(country: str):
    """국가별 누적 수익률"""
    sql = f"""
    SELECT trade_date, benchmark_return, first_return, fresh_return, smart_return
    FROM country_cumulative_returns
    WHERE date = (SELECT MAX(date) FROM country_cumulative_returns)
    AND country = '{country}'
    ORDER BY trade_date
    """
    return query_df(sql)

# =============================================================================
# Live API Functions (for time-series charts)
# =============================================================================
@st.cache_data(ttl=3600)
def load_vix_history(days=180):
    try:
        start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        vix = yf.download('^VIX', start=start, progress=False)
        if vix.empty:
            return pd.DataFrame()
        if isinstance(vix.columns, pd.MultiIndex):
            vix = vix['Close'].iloc[:, 0] if 'Close' in vix.columns.get_level_values(0) else vix.iloc[:, 0]
        elif 'Close' in vix.columns:
            vix = vix['Close']
        else:
            vix = vix.iloc[:, 0]
        return vix.to_frame(name='VIX')
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_dxy_history(days=180):
    try:
        start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        dxy = yf.download('DX-Y.NYB', start=start, progress=False)
        if dxy.empty:
            return pd.DataFrame()
        if isinstance(dxy.columns, pd.MultiIndex):
            dxy = dxy['Close'].iloc[:, 0] if 'Close' in dxy.columns.get_level_values(0) else dxy.iloc[:, 0]
        elif 'Close' in dxy.columns:
            dxy = dxy['Close']
        else:
            dxy = dxy.iloc[:, 0]
        return dxy.to_frame(name='DXY')
    except:
        return pd.DataFrame()

# =============================================================================
# Visualization Functions (Inline - Git에 포함됨)
# =============================================================================
def create_indicator_gauge(value, title, min_val, max_val, thresholds=None, reverse_colors=False):
    """게이지 차트 생성"""
    if value is None or pd.isna(value):
        value = (min_val + max_val) / 2
    value = max(min_val, min(max_val, float(value)))
    
    if thresholds is None:
        low, high = min_val + (max_val - min_val) * 0.33, min_val + (max_val - min_val) * 0.66
    else:
        low, high = thresholds.get('low', 25), thresholds.get('high', 75)
    
    if reverse_colors:
        steps = [
            {'range': [min_val, low], 'color': "#f8d7da"},
            {'range': [low, high], 'color': "#fff3cd"},
            {'range': [high, max_val], 'color': "#d4edda"}
        ]
    else:
        steps = [
            {'range': [min_val, low], 'color': "#d4edda"},
            {'range': [low, high], 'color': "#fff3cd"},
            {'range': [high, max_val], 'color': "#f8d7da"}
        ]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={'text': title, 'font': {'size': 16}},
        gauge={'axis': {'range': [min_val, max_val]}, 'bar': {'color': "darkblue"},
               'steps': steps, 'threshold': {'line': {'color': "red", 'width': 2}, 'thickness': 0.75, 'value': value}}
    ))
    fig.update_layout(height=250, margin=dict(t=50, b=30, l=30, r=30))
    return fig

def create_sector_heatmap(sector_df):
    """섹터 히트맵 생성"""
    if sector_df.empty:
        return go.Figure()
    sectors = sector_df['sector'].tolist()
    returns_1d = sector_df['return_1d'].tolist()
    
    fig = go.Figure(data=go.Heatmap(
        z=[returns_1d], x=sectors, y=['1D Return'],
        colorscale='RdYlGn', zmid=0,
        text=[[f"{r:.2%}" if pd.notna(r) else "-" for r in returns_1d]],
        texttemplate="%{text}", textfont={"size": 12},
        hovertemplate='%{x}: %{z:.2%}<extra></extra>'
    ))
    fig.update_layout(height=120, margin=dict(t=20, b=20, l=20, r=20))
    return fig

def plot_business_clock(df, title, compare=False):
    """Business Cycle Clock (4분면 차트)"""
    fig = go.Figure()
    
    if df is None or df.empty:
        fig.update_layout(title=title, height=380)
        return fig
    
    # 축 범위 계산
    axis_range = 2
    if 'cli_level' in df.columns and 'cli_momentum' in df.columns:
        df_valid = df.dropna(subset=['cli_level', 'cli_momentum'])
        if not df_valid.empty:
            max_val = max(abs(df_valid['cli_level']).max(), abs(df_valid['cli_momentum']).max())
            axis_range = max(max_val * 1.2, 0.5)
    
    label_pos = axis_range * 0.7
    
    # 4분면 배경
    fig.add_shape(type="rect", x0=0, x1=axis_range, y0=0, y1=axis_range,
                  fillcolor="rgba(44, 160, 44, 0.25)", line=dict(width=0), layer='below')
    fig.add_shape(type="rect", x0=-axis_range, x1=0, y0=0, y1=axis_range,
                  fillcolor="rgba(255, 249, 196, 0.4)", line=dict(width=0), layer='below')
    fig.add_shape(type="rect", x0=-axis_range, x1=0, y0=-axis_range, y1=0,
                  fillcolor="rgba(255, 230, 230, 0.4)", line=dict(width=0), layer='below')
    fig.add_shape(type="rect", x0=0, x1=axis_range, y0=-axis_range, y1=0,
                  fillcolor="rgba(255, 243, 224, 0.4)", line=dict(width=0), layer='below')
    
    fig.add_hline(y=0, line_color="gray", line_width=1, line_dash="dot")
    fig.add_vline(x=0, line_color="gray", line_width=1, line_dash="dot")
    
    if 'cli_level' in df.columns and 'cli_momentum' in df.columns:
        d = df.dropna(subset=['cli_level', 'cli_momentum'])
        x, y = d['cli_level'].values, d['cli_momentum'].values
        
        # Compare 모드
        if compare and 'level_first' in d.columns and 'momentum_first' in d.columns:
            valid_first = d.dropna(subset=['level_first', 'momentum_first'])
            if not valid_first.empty:
                fig.add_trace(go.Scatter(
                    x=valid_first['level_first'], y=valid_first['momentum_first'],
                    mode='markers', marker=dict(size=5, color='gray', opacity=0.5),
                    name='First Value'
                ))
        
        # 경로 라인
        if len(x) > 1:
            for i in range(len(x) - 1):
                color_intensity = int(50 + (i / (len(x) - 1)) * 150)
                color = f'rgb({50}, {50 + color_intensity//2}, {color_intensity + 100})'
                fig.add_trace(go.Scatter(
                    x=[x[i], x[i+1]], y=[y[i], y[i+1]],
                    mode='lines', line=dict(color=color, width=3),
                    showlegend=False, hoverinfo='skip'
                ))
        
        # 경로 포인트
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='markers',
            marker=dict(size=10, color='white', line=dict(color='navy', width=2)),
            name='Path'
        ))
        
        # 최신 포인트
        if len(x) > 0:
            fig.add_trace(go.Scatter(
                x=[x[-1]], y=[y[-1]], mode='markers',
                marker=dict(size=14, color='red', line=dict(color='white', width=2)),
                name='Latest'
            ))
    
    # 라벨
    fig.add_annotation(x=label_pos, y=label_pos, text="<b>Expansion</b>", showarrow=False, font=dict(size=12, color='green'))
    fig.add_annotation(x=-label_pos, y=label_pos, text="<b>Recovery</b>", showarrow=False, font=dict(size=12, color='goldenrod'))
    fig.add_annotation(x=-label_pos, y=-label_pos, text="<b>Contraction</b>", showarrow=False, font=dict(size=12, color='red'))
    fig.add_annotation(x=label_pos, y=-label_pos, text="<b>Slowdown</b>", showarrow=False, font=dict(size=12, color='darkorange'))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        xaxis=dict(range=[-axis_range, axis_range], zeroline=False, showgrid=False, title="Level", scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-axis_range, axis_range], zeroline=False, showgrid=False, title="Momentum"),
        height=380, margin=dict(l=50, r=50, t=50, b=50), showlegend=False, plot_bgcolor='white'
    )
    return fig

def plot_regime_strip(df):
    """국면 스트립 차트 (Timeline)"""
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(title="국면 스트립 차트 - 데이터 없음", height=200)
        return fig
    
    timeline_data = []
    for method_name, method_col in [('FIRST', 'exp1_regime'), ('FRESH', 'exp2_regime'), ('SMART', 'exp3_regime')]:
        if method_col not in df.columns:
            continue
        sub = df[['trade_date', method_col]].dropna(subset=[method_col])
        if sub.empty:
            continue
        sub['next_date'] = sub['trade_date'].shift(-1).fillna(pd.Timestamp.now())
        for _, row in sub.iterrows():
            timeline_data.append({
                'Task': method_name, 'Start': row['trade_date'], 'Finish': row['next_date'],
                'Regime': row[method_col], 'Color': REGIME_COLORS.get(row[method_col], '#cccccc')
            })
    
    if not timeline_data:
        fig = go.Figure()
        fig.update_layout(title="국면 스트립 차트 - 데이터 없음", height=200)
        return fig
    
    df_timeline = pd.DataFrame(timeline_data)
    fig = px.timeline(df_timeline, x_start='Start', x_end='Finish', y='Task', color='Regime',
                      color_discrete_map=REGIME_COLORS)
    fig.update_layout(title="국면 스트립 차트", height=250, showlegend=True,
                      yaxis=dict(categoryorder='array', categoryarray=['SMART', 'FRESH', 'FIRST']))
    return fig

def plot_country_cumulative_returns(df, country_name):
    """국가별 누적 수익률 차트"""
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"[{country_name}] 데이터 없음")
        return fig
    
    fig = go.Figure()
    if 'benchmark_return' in df.columns:
        fig.add_trace(go.Scatter(x=df['trade_date'], y=df['benchmark_return'], name='Benchmark', line=dict(color='silver', dash='dash')))
    if 'first_return' in df.columns:
        fig.add_trace(go.Scatter(x=df['trade_date'], y=df['first_return'], name='First', line=dict(color='#1f77b4', width=2)))
    if 'fresh_return' in df.columns:
        fig.add_trace(go.Scatter(x=df['trade_date'], y=df['fresh_return'], name='Fresh', line=dict(color='#2ca02c', width=2)))
    if 'smart_return' in df.columns:
        fig.add_trace(go.Scatter(x=df['trade_date'], y=df['smart_return'], name='Smart', line=dict(color='#d62728', width=2)))
    
    fig.update_layout(title=f"[{country_name}] 누적 수익률", height=350,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      yaxis_tickformat='.0%', hovermode='x unified')
    return fig

# =============================================================================
# Page Config
# =============================================================================
st.set_page_config(page_title="Market Regime Dashboard", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    @media (max-width: 768px) { .main-header { font-size: 1.5rem; } }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Sidebar
# =============================================================================
st.sidebar.title("⚙️ Settings")
if st.sidebar.button("🔄 데이터 새로고침"):
    st.cache_data.clear()
    st.rerun()

# =============================================================================
# Main Content
# =============================================================================
st.markdown('<div class="main-header">📊 Market Regime Dashboard</div>', unsafe_allow_html=True)

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
        for col in ['1D', '1W', '1M', 'YTD', '1Y']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
        display_df['종가'] = display_df['종가'].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "-")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("지수 데이터 없음")

with idx_tab2:
    sector_df = load_sector_returns()
    if not sector_df.empty:
        fig_heatmap = create_sector_heatmap(sector_df)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
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
    colors = {'팽창': 'background-color: #2ca02c; color: white', '회복': 'background-color: #ffce30; color: black',
              '둔화': 'background-color: #ff7f0e; color: white', '침체': 'background-color: #d62728; color: white',
              'Cash': 'background-color: #ffb347; color: black', 'Half': 'background-color: #9467bd; color: white',
              'Skipped': 'background-color: #f0f0f0; color: black'}
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

weights_df = load_portfolio_weights()
metrics_df = load_backtest_metrics()
cum_returns_df = load_cumulative_returns()

if not weights_df.empty:
    st.markdown("#### 📍 현재 포지션")
    pos_df = weights_df[weights_df['weight'] > 0.001][['country', 'ticker', 'regime', 'weight']].copy()
    pos_df['weight'] = pos_df['weight'].apply(lambda x: f"{x:.1%}")
    pos_df.columns = ['국가', 'Ticker', 'Regime', '비중']
    
    def color_regime_pos(val):
        colors = {'팽창': 'background-color: #2ca02c; color: white', '회복': 'background-color: #ffce30; color: black',
                  '둔화': 'background-color: #ff7f0e; color: white', '침체': 'background-color: #d62728; color: white'}
        return colors.get(val, '')
    
    styled_pos = pos_df.style.applymap(color_regime_pos, subset=['Regime'])
    st.dataframe(styled_pos, hide_index=True, use_container_width=True)

# 누적수익률 차트
if not cum_returns_df.empty:
    st.markdown("#### 📈 누적 수익률 (Backtest)")
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=cum_returns_df['date'], y=cum_returns_df['strategy_return'],
                                  name='Strategy', line=dict(color='#2ca02c', width=2)))
    fig_cum.add_trace(go.Scatter(x=cum_returns_df['date'], y=cum_returns_df['benchmark_return'],
                                  name='Benchmark (ACWI)', line=dict(color='silver', width=2, dash='dash')))
    fig_cum.update_layout(height=350, yaxis_tickformat='.0%', hovermode='x unified',
                          legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    st.plotly_chart(fig_cum, use_container_width=True)

# 성과 지표
if not metrics_df.empty:
    m = metrics_df.iloc[0]
    perf_cols = st.columns(4)
    with perf_cols[0]:
        st.metric("CAGR", f"{m['cagr']:.1%}" if pd.notna(m['cagr']) else "-", 
                  delta=f"{(m['cagr'] - m['bm_cagr'])*100:.1f}%p" if pd.notna(m['bm_cagr']) else None)
    with perf_cols[1]:
        st.metric("Sharpe", f"{m['sharpe']:.2f}" if pd.notna(m['sharpe']) else "-")
    with perf_cols[2]:
        st.metric("MDD", f"{m['mdd']:.1%}" if pd.notna(m['mdd']) else "-")
    with perf_cols[3]:
        st.metric("Vol", f"{m['volatility']:.1%}" if pd.notna(m['volatility']) else "-")

# 리밸런싱 이력
with st.expander("📅 리밸런싱 이력 (최근)"):
    rebal_df = load_rebalancing_history()
    if not rebal_df.empty:
        pivot_df = rebal_df.pivot(index='trade_date', columns='country', values='weight')
        pivot_df = (pivot_df * 100).round(1).tail(10)
        st.dataframe(pivot_df, use_container_width=True)
    else:
        st.info("리밸런싱 이력 없음")

st.markdown("---")

# =============================================================================
# Country Detail Tabs (국가별 상세 분석)
# =============================================================================
st.subheader("📈 국가별 상세 분석")

countries_in_db = regime_df['country'].tolist()
if countries_in_db:
    tabs = st.tabs([f"🏳️ {COUNTRY_MAP.get(c, {}).get('name', c)}" for c in countries_in_db])
    
    for i, country in enumerate(countries_in_db):
        with tabs[i]:
            info = COUNTRY_MAP.get(country, {'name': country, 'ticker': ''})
            
            # 데이터 로딩
            country_regimes = load_country_regimes(country)
            country_returns = load_country_cumulative_returns(country)
            
            if country_regimes.empty and country_returns.empty:
                st.info(f"{country}: 상세 데이터 없음 (data_exporter 실행 필요)")
                continue
            
            # 누적 수익률 차트
            if not country_returns.empty:
                st.markdown("#### 📊 누적 수익률")
                fig_returns = plot_country_cumulative_returns(country_returns, info['name'])
                st.plotly_chart(fig_returns, use_container_width=True)
            
            # 국면 스트립 차트
            if not country_regimes.empty:
                st.markdown("#### 📅 국면 타임라인")
                fig_strip = plot_regime_strip(country_regimes)
                st.plotly_chart(fig_strip, use_container_width=True)
            
            # Business Cycle Clock
            if not country_regimes.empty:
                st.markdown("#### 🕐 Business Cycle Clock")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_clock = plot_business_clock(country_regimes.tail(24), "PIT History", compare=True)
                    st.plotly_chart(fig_clock, use_container_width=True)
                
                with col2:
                    # 최근 3개월만 표시
                    fig_clock_recent = plot_business_clock(country_regimes.tail(6), "Recent (6mo)", compare=False)
                    st.plotly_chart(fig_clock_recent, use_container_width=True)

st.markdown("---")

# =============================================================================
# 시장 심리 (하단)
# =============================================================================
st.subheader("💭 시장 심리")

indicators_df = load_market_indicators()

fg_col1, fg_col2, fg_col3 = st.columns([1, 2, 1])
with fg_col2:
    if not indicators_df.empty and pd.notna(indicators_df.iloc[0]['fear_greed']):
        fg_val = indicators_df.iloc[0]['fear_greed']
        fg_text = indicators_df.iloc[0]['fear_greed_text']
        fig_fg = create_indicator_gauge(fg_val, "CNN Fear & Greed Index", 0, 100,
                                          thresholds={'low': 25, 'high': 75}, reverse_colors=True)
        st.plotly_chart(fig_fg, use_container_width=True, config={'displayModeBar': False})
        if fg_text:
            st.markdown(f"<center><b>{fg_text}</b></center>", unsafe_allow_html=True)
    else:
        st.info("Fear & Greed 데이터 없음")

# VIX/DXY 시계열 차트
with st.expander("📈 VIX / DXY 시계열"):
    vix_col, dxy_col = st.columns(2)
    
    with vix_col:
        vix_hist = load_vix_history(180)
        if not vix_hist.empty:
            fig_vix = go.Figure()
            fig_vix.add_trace(go.Scatter(x=vix_hist.index, y=vix_hist['VIX'], name='VIX', line=dict(color='#d62728', width=2)))
            fig_vix.update_layout(title='VIX (180D)', height=250, margin=dict(t=40, b=30))
            st.plotly_chart(fig_vix, use_container_width=True)
        else:
            st.info("VIX 데이터 로딩 실패")
    
    with dxy_col:
        dxy_hist = load_dxy_history(180)
        if not dxy_hist.empty:
            fig_dxy = go.Figure()
            fig_dxy.add_trace(go.Scatter(x=dxy_hist.index, y=dxy_hist['DXY'], name='DXY', line=dict(color='#1f77b4', width=2)))
            fig_dxy.update_layout(title='DXY (180D)', height=250, margin=dict(t=40, b=30))
            st.plotly_chart(fig_dxy, use_container_width=True)
        else:
            st.info("DXY 데이터 로딩 실패")

# =============================================================================
# Sidebar - Market Indicators
# =============================================================================
st.sidebar.title("📊 Market Indicators")
if not indicators_df.empty:
    m = indicators_df.iloc[0]
    if pd.notna(m['fear_greed']):
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

# =============================================================================
# Footer
# =============================================================================
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: gray; font-size: 0.8rem;">
    📊 Market Regime Dashboard | Data Source: MariaDB, Yahoo Finance | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>
""", unsafe_allow_html=True)
