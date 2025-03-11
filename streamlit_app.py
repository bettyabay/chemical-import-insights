# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Chemical Imports, Freight Forwarding, and Logistics",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling
st.markdown("""
<style>
[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}
[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}
</style>
""", unsafe_allow_html=True)

# Load data
data_import = pd.read_csv('data/cleaned/cleaned_import_data.csv')

# Sidebar
with st.sidebar:
    st.title('🏂 Chemical Imports, Freight Forwarding, and Logistics Dashboard')
    
    # Extract year and month for selection
    data_import['Reg. date'] = pd.to_datetime(data_import['Reg. date'])
    year_list = data_import['Reg. date'].dt.year.unique()[::-1]
    selected_year = st.selectbox('Select a year', year_list)
    month_list = data_import['Reg. date'].dt.month.unique()
    selected_month = st.selectbox('Select a month', month_list)

    # Filter data based on selections
    df_selected = data_import[(data_import['Reg. date'].dt.year == selected_year) & 
                              (data_import['Reg. date'].dt.month == selected_month)]

    # Color theme selection
    color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
    selected_color_theme = st.selectbox('Select a color theme', color_theme_list)

# Plots
def make_heatmap(input_df, input_y, input_x, input_color, input_color_theme):
    heatmap = alt.Chart(input_df).mark_rect().encode(
        y=alt.Y(f'{input_y}:O', axis=alt.Axis(title="Trader")),
        x=alt.X(f'{input_x}:O', axis=alt.Axis(title="# of packages")),
        color=alt.Color(f'sum({input_color}):Q',
                         legend=None,
                         scale=alt.Scale(scheme=input_color_theme)),
        stroke=alt.value('black'),
        strokeWidth=alt.value(0.25),
    ).properties(width=900).configure_axis(
        labelFontSize=12,
        titleFontSize=12
    )
    return heatmap

# Display total tax paid by each trader
trader_tax_amount = data_import.groupby('Trader')['# of packages'].sum().reset_index()
st.write("Total number of packages imported by Each Trader:")
st.dataframe(trader_tax_amount)

# Class distribution visualization
class_distribution = data_import['Commercial / Brand Name'].value_counts()
st.write("Class Distribution:")
st.bar_chart(class_distribution)

# Display selected data
if df_selected.shape[0] > 0:
    st.write("Selected Data:")
    st.dataframe(df_selected)

    # Heatmap of the selected month
    heatmap = make_heatmap(df_selected, 'Trader', '# of packages', '# of packages', selected_color_theme)
    st.altair_chart(heatmap, use_container_width=True)
else:
    st.write("No data available for the selected year and month.")

# Provide options to download aggregate data
if st.button('Download Trader Category Summary'):
    trader_category_summary = df_selected.groupby(['Trader', 'Commercial / Brand Name']).size().reset_index(name='Count')
    trader_category_summary.to_csv('../data/cleaned/trader_category_summary.csv', index=False)
    st.success('Trader Category Summary downloaded successfully!')

# About section
with st.expander('About', expanded=True):
    st.write('''
        - Insights on traders, number of packages, and product categories.
    ''')