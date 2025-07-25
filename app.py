#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder

# Set page config
st.set_page_config(
    page_title="EV Adoption Forecast",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load necessary data and models
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_resource
def load_model():
    model = joblib.load('forecasting_ev_model.pkl')
    return model

@st.cache_resource
def load_encoder():
    df = load_data()
    le = LabelEncoder()
    le.fit(df['County'])
    return le

# Load resources
df = load_data()
model = load_model()
le = load_encoder()

# Define features (must match training)
features = [
    'months_since_start',
    'county_encoded',
    'ev_total_lag1',
    'ev_total_lag2',
    'ev_total_lag3',
    'ev_total_roll_mean_3',
    'ev_total_pct_change_1',
    'ev_total_pct_change_3',
    'ev_growth_slope',
]

# Sidebar controls
st.sidebar.title("Forecast Settings")
selected_county = st.sidebar.selectbox(
    "Select County",
    sorted(df['County'].unique()),
    index=0
)

forecast_months = st.sidebar.slider(
    "Forecast Horizon (months)",
    1, 36, 12
)

show_confidence = st.sidebar.checkbox(
    "Show Confidence Interval",
    value=True
)

# Main content
st.title("🚗 Electric Vehicle Adoption Forecast")
st.markdown("""
This app forecasts future EV adoption based on historical trends in Washington State.
""")

# Get county data
county_code = le.transform([selected_county])[0]
county_df = df[df['county_encoded'] == county_code].sort_values("Date")

# Show historical summary
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total EVs (Historical)", f"{county_df['Electric Vehicle (EV) Total'].sum():,}")
with col2:
    st.metric("Current EV Count", f"{county_df['Electric Vehicle (EV) Total'].iloc[-1]:,}")
with col3:
    st.metric("EV Percentage", f"{county_df['Percent Electric Vehicles'].iloc[-1]:.1%}")

# Generate forecast
if st.sidebar.button("Generate Forecast"):
    with st.spinner("Generating forecast..."):
        # Prepare historical data
        historical_ev = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
        cumulative_ev = list(np.cumsum(historical_ev))
        months_since_start = county_df['months_since_start'].max()
        
        forecast_dates = []
        forecast_values = []
        lower_bounds = []
        upper_bounds = []
        
        for i in range(forecast_months):
            # Prepare features
            lag1, lag2, lag3 = historical_ev[-1], historical_ev[-2], historical_ev[-3]
            roll_mean = np.mean([lag1, lag2, lag3])
            pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
            pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
            
            recent_cumulative = cumulative_ev[-6:]
            ev_growth_slope = np.polyfit(range(len(recent_cumulative)), recent_cumulative, 1)[0] if len(recent_cumulative) == 6 else 0

            # Create prediction row
            new_row = {
                'months_since_start': months_since_start + i + 1,
                'county_encoded': county_code,
                'ev_total_lag1': lag1,
                'ev_total_lag2': lag2,
                'ev_total_lag3': lag3,
                'ev_total_roll_mean_3': roll_mean,
                'ev_total_pct_change_1': pct_change_1,
                'ev_total_pct_change_3': pct_change_3,
                'ev_growth_slope': ev_growth_slope
            }
            
            # Predict
            X_new = pd.DataFrame([new_row])[features]
            pred = model.predict(X_new)[0]
            
            # Confidence interval (simplified)
            lower = pred * 0.9
            upper = pred * 1.1
            
            forecast_date = county_df['Date'].max() + timedelta(days=30*(i+1))
            forecast_dates.append(forecast_date)
            forecast_values.append(pred)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
            
            # Update rolling window
            historical_ev.append(pred)
            historical_ev = historical_ev[-6:]
            cumulative_ev.append(cumulative_ev[-1] + pred)
            cumulative_ev = cumulative_ev[-6:]

        # Create DataFrames for plotting
        historical_df = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
        historical_df['Type'] = 'Historical'
        
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Electric Vehicle (EV) Total': forecast_values,
            'Lower_CI': lower_bounds,
            'Upper_CI': upper_bounds,
            'Type': 'Forecast'
        })
        
        combined_df = pd.concat([
            historical_df,
            forecast_df[['Date', 'Electric Vehicle (EV) Total', 'Type']]
        ])
        
        # Plot monthly forecast
        st.subheader(f"Monthly EV Forecast for {selected_county} County")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot historical
        ax.plot(
            historical_df['Date'],
            historical_df['Electric Vehicle (EV) Total'],
            label='Historical',
            color='blue',
            marker='o'
        )
        
        # Plot forecast
        ax.plot(
            forecast_df['Date'],
            forecast_df['Electric Vehicle (EV) Total'],
            label='Forecast',
            color='orange',
            linestyle='--',
            marker='o'
        )
        
        # Plot confidence interval
        if show_confidence:
            ax.fill_between(
                forecast_df['Date'],
                forecast_df['Lower_CI'],
                forecast_df['Upper_CI'],
                color='orange',
                alpha=0.2,
                label='90% Confidence'
            )
        
        ax.set_title(f"EV Adoption Forecast - {selected_county} County")
        ax.set_xlabel("Date")
        ax.set_ylabel("EV Count")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
        
        # Cumulative view
        st.subheader("Cumulative EV Adoption")
        cumulative_df = combined_df.copy()
        cumulative_df['Cumulative EVs'] = cumulative_df['Electric Vehicle (EV) Total'].cumsum()
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(
            cumulative_df['Date'],
            cumulative_df['Cumulative EVs'],
            label='Cumulative EVs',
            color='green'
        )
        
        # Mark forecast start point
        forecast_start = forecast_df['Date'].min()
        ax2.axvline(
            x=forecast_start,
            color='red',
            linestyle='--',
            label='Forecast Start'
        )
        
        ax2.set_title(f"Cumulative EV Adoption - {selected_county} County")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Total EVs")
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig2)
        
        # Show forecast summary
        st.subheader("Forecast Summary")
        forecast_summary = forecast_df.copy()
        forecast_summary['Year'] = forecast_summary['Date'].dt.year
        forecast_summary['Month'] = forecast_summary['Date'].dt.month_name()
        
        yearly_summary = forecast_summary.groupby('Year').agg({
            'Electric Vehicle (EV) Total': 'sum'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Yearly Forecast Summary**")
            st.dataframe(yearly_summary.style.format({
                'Electric Vehicle (EV) Total': '{:,.0f}'
            }))
        
        with col2:
            st.write("**Forecast Details**")
            st.dataframe(forecast_summary[['Year', 'Month', 'Electric Vehicle (EV) Total']].style.format({
                'Electric Vehicle (EV) Total': '{:,.0f}'
            }))

# Add some info
st.sidebar.markdown("---")
st.sidebar.info("""
**About this app:**
- Uses XGBoost model trained on WA state EV data
- Forecasts based on historical trends and county patterns
- Confidence intervals are estimates (±10%)
""")