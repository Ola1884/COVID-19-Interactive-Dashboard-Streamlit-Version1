# Import the necessary libraries
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from model_utils import get_available_countries, get_country_data, get_global_data, prophet_forecast, get_model_comparison, random_forest_forecast

# Set page configuration
st.set_page_config(
    page_title="COVID-19 Interactive Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #ffffff;
        background-color: #2c3e50;
        padding: 20px;
        text-align: center;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #e74c3c;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: white;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #c0392b;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">COVID-19 Interactive Dashboard</h1>', unsafe_allow_html=True)

# Sidebar for country selection
st.sidebar.header("Country Selection")
country_options = get_available_countries()
default_country = 'US' if 'US' in country_options else country_options[0]
selected_country = st.sidebar.selectbox("Select a country", country_options, index=country_options.index(default_country))

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Forecast & Trends", 
    "🌍 World Map", 
    "📊 Data Table", 
    "💡 Insights", 
    "🤖 Model Comparison"
])

# Forecast & Trends Tab
with tab1:
    st.header(f"COVID-19 Forecast for {selected_country}")
    
    # Get historical data AND forecast data
    historical_df = get_country_data(selected_country)
    prophet_df, future_dates, forecast_values, _ = prophet_forecast(selected_country)
    
    # Create forecast plot
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=historical_df['Date'],
        y=historical_df['Cases'],
        mode='lines',
        name='Historical Data',
        line=dict(color='blue')
    ))
    fig_forecast.add_trace(go.Scatter(
        x=future_dates,
        y=forecast_values,
        mode='lines',
        name='Prophet Forecast',
        line=dict(color='red', dash='dash')
    ))
    fig_forecast.update_layout(
        title=f'COVID-19 Forecast for {selected_country}',
        xaxis_title='Date', 
        yaxis_title='Cases',
        height=500
    )
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Create trends plot
    fig_trends = go.Figure()
    fig_trends.add_trace(go.Scatter(
        x=historical_df['Date'], 
        y=historical_df['Cases'],
        mode='lines', 
        name='Actual Cases', 
        line=dict(color='blue')
    ))
    fig_trends.add_trace(go.Scatter(
        x=historical_df['Date'], 
        y=historical_df['7Day_MA'],
        mode='lines', 
        name='7-Day Average', 
        line=dict(color='green', dash='dot')
    ))
    fig_trends.update_layout(
        title=f'Trend Analysis for {selected_country}',
        xaxis_title='Date', 
        yaxis_title='Cases',
        height=500
    )
    st.plotly_chart(fig_trends, use_container_width=True)

# World Map Tab
with tab2:
    st.header("Global COVID-19 Cases Distribution")
    
    map_df = get_global_data()
    map_fig = px.choropleth(
        map_df, 
        locations="Country", 
        locationmode='country names',
        color="Cases", 
        hover_name="Country",
        hover_data=["Cases"],
        color_continuous_scale="reds",
        title="Global COVID-19 Cases Distribution"
    )
    map_fig.update_layout(geo=dict(showframe=False, showcoastlines=False), height=600)
    st.plotly_chart(map_fig, use_container_width=True)

# Data Table Tab
with tab3:
    st.header(f"30-Day Forecast for {selected_country}")
    
    prophet_df, future_dates, forecast_values, prophet_metrics = prophet_forecast(selected_country)
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Cases': forecast_values.round().astype(int)
    })
    
    # Display as a table
    st.dataframe(
        forecast_df,
        column_config={
            "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            "Predicted Cases": st.column_config.NumberColumn("Predicted Cases", format="%d")
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Also show as a chart for better visualization
    fig_table = go.Figure(data=[go.Table(
        header=dict(
            values=['Date', 'Predicted Cases'], 
            fill_color='paleturquoise', 
            align='left'
        ),
        cells=dict(
            values=[forecast_df['Date'].dt.strftime('%Y-%m-%d'), forecast_df['Predicted Cases']],
            fill_color='lavender', 
            align='left'
        ))
    ])
    fig_table.update_layout(title=f'30-Day Forecast for {selected_country}', height=400)
    st.plotly_chart(fig_table, use_container_width=True)

# Insights Tab
with tab4:
    st.header("📊 COVID-19 Insights")
    
    historical_df = get_country_data(selected_country)
    prophet_df, future_dates, forecast_values, prophet_metrics = prophet_forecast(selected_country)
    
    # Calculate metrics
    last_historical_value = historical_df['Cases'].iloc[-1]
    forecast_value_30d = forecast_values.iloc[-1] if len(forecast_values) > 0 else 0
    forecast_growth = ((forecast_value_30d - last_historical_value) / last_historical_value * 100) if last_historical_value > 0 else 0
    
    # Current trend analysis
    current_ma = historical_df['7Day_MA'].iloc[-1]
    previous_ma = historical_df['7Day_MA'].iloc[-8] if len(historical_df) > 7 else current_ma
    trend = "Increasing" if current_ma > previous_ma else "Decreasing"
    trend_percentage = abs((current_ma - previous_ma) / previous_ma * 100) if previous_ma > 0 else 0
    
    # Create metrics columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Cases", f"{last_historical_value:,.0f}")
        st.metric("Peak Cases", f"{historical_df['Cases'].max():,.0f}")
    
    with col2:
        st.metric("30-Day Forecast", f"{forecast_value_30d:,.0f}")
        st.metric("Projected Growth", f"{forecast_growth:+.1f}%")
    
    with col3:
        st.metric("7-Day Trend", trend, f"{trend_percentage:.1f}%")
        st.metric("Data Duration", f"{len(historical_df)} days")
    
    # Additional insights
    st.subheader("Additional Information")
    st.info("""
    - **Current Cases**: Latest available case count
    - **30-Day Forecast**: Projected cases in 30 days using Prophet model
    - **Projected Growth**: Expected growth rate over 30 days
    - **7-Day Trend**: Current trend based on 7-day moving average
    - **Peak Cases**: Highest recorded cases
    - **Data Duration**: Available historical data points
    """)

# Model Comparison Tab
with tab5:
    st.header(f"Model Comparison for {selected_country}")
    
    comparison = get_model_comparison(selected_country)
    
    # Create comparison plot
    fig_comparison = go.Figure()
    fig_comparison.add_trace(go.Scatter(
        x=comparison['historical']['Date'], 
        y=comparison['historical']['Cases'],
        mode='lines', 
        name='Historical Data', 
        line=dict(color='blue', width=2)
    ))
    fig_comparison.add_trace(go.Scatter(
        x=comparison['prophet']['dates'], 
        y=comparison['prophet']['values'],
        mode='lines+markers', 
        name='Prophet Forecast', 
        line=dict(color='red', dash='dash')
    ))
    fig_comparison.add_trace(go.Scatter(
        x=comparison['random_forest']['dates'], 
        y=comparison['random_forest']['values'],
        mode='lines+markers', 
        name='Random Forest Forecast', 
        line=dict(color='green', dash='dot')
    ))
    
    fig_comparison.update_layout(
        title=f'Model Comparison for {selected_country}',
        xaxis_title='Date',
        yaxis_title='Cases',
        hovermode='x unified',
        height=500
    )
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Model metrics
    st.subheader("Model Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 🤖 Random Forest:")
        st.metric("Mean Absolute Error (MAE)", f"{comparison['random_forest']['metrics']['mae']:.2f}")
        st.metric("Root Mean Squared Error (RMSE)", f"{comparison['random_forest']['metrics']['rmse']:.2f}")
    
    with col2:
        st.markdown("##### 🔮 Prophet:")
        st.metric("Mean Absolute Error (MAE)", f"{comparison['prophet']['metrics']['mae']:.2f}")
        st.metric("Root Mean Squared Error (RMSE)", f"{comparison['prophet']['metrics']['rmse']:.2f}")
