# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

#######################
# Page configuration
st.set_page_config(
    page_title="Chemical Imports, Freight Forwarding, and Logistics",
    page_icon="🏂",
    layout="wide",
    initial_sidebar_state="expanded"
)

alt.themes.enable("dark")

#######################
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

#######################
# Load data
data_import = pd.read_csv('C:/Users/USER/Desktop/chemical-import-insights/data/cleaned/data_model_for_trainig.csv')

#######################
# Sidebar
with st.sidebar:
    st.title('🏂 Chemical Imports, Freight Forwarding, and Logistics Dashboard')
    
    year_list = data_import['Reg. date'].apply(lambda x: x.split('-')[0]).unique()[::-1]
    selected_year = st.selectbox('Select a year', year_list)
    
    # Filter by Trader
    trader_list = data_import['Trader'].unique()
    selected_trader = st.selectbox('Select a Trader', trader_list)
    
    # Month filter
    month_list = data_import['Reg. date'].apply(lambda x: x.split('-')[1]).unique()
    selected_month = st.selectbox('Select a Month', month_list)
    
    color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
    selected_color_theme = st.selectbox('Select a color theme', color_theme_list)

#######################
# Filter Data
df_selected_year = data_import[data_import['Reg. date'].str.startswith(selected_year)]
df_selected_trader = df_selected_year[df_selected_year['Trader'] == selected_trader]
df_selected_month = df_selected_trader[df_selected_trader['Reg. date'].str.contains(f'-{selected_month}-')]

#######################
# Plots

# Heatmap
def make_heatmap(input_df, input_y, input_x, input_color, input_color_theme):
    heatmap = alt.Chart(input_df).mark_rect().encode(
        y=alt.Y(f'{input_y}:O', axis=alt.Axis(title="Month", titleFontSize=18, titlePadding=15, titleFontWeight=900, labelAngle=0)),
        x=alt.X(f'{input_x}:O', axis=alt.Axis(title="", titleFontSize=18, titlePadding=15, titleFontWeight=900)),
        color=alt.Color(f'max({input_color}):Q',
                         legend=None,
                         scale=alt.Scale(scheme=input_color_theme)),
        stroke=alt.value('black'),
        strokeWidth=alt.value(0.25),
    ).properties(width=900).configure_axis(
        labelFontSize=12,
        titleFontSize=12
    )
    return heatmap

# Choropleth map
def make_choropleth(input_df, input_id, input_column, input_color_theme):
    choropleth = px.choropleth(input_df, locations=input_id, color=input_column, locationmode="USA-states",
                                color_continuous_scale=input_color_theme,
                                range_color=(0, input_df[input_column].max()),
                                scope="usa",
                                labels={input_column: 'Value'})
    choropleth.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=350
    )
    return choropleth

# Dashboard Main Panel
col = st.columns((1.5, 4.5, 2), gap='medium')

with col[0]:
    st.markdown('#### Gains/Losses')
    # Here you can add calculations for gains/losses if needed
    
    st.markdown('#### States Migration')
    # You can add migration calculations here as necessary

with col[1]:
    st.markdown('#### Trader')
    
    # Trader is a relevant column
    choropleth = make_choropleth(df_selected_month, 'Trader', '# of packages', selected_color_theme)
    st.plotly_chart(choropleth, use_container_width=True)

    # Heatmap for the number of packages by month
    df_heatmap = df_selected_month.groupby(['Reg. date', '# of packages']).sum().reset_index()
    heatmap = make_heatmap(df_heatmap, 'Reg. date', '# of packages', 'Commercial / Brand Name', selected_color_theme)
    st.altair_chart(heatmap, use_container_width=True)

with col[2]:
    st.markdown('#### Top Traders')
    
    df_top_traders = df_selected_month.groupby('Trader')['# of packages'].sum().nlargest(10).reset_index()
    st.dataframe(df_top_traders)
    
    with st.expander('About', expanded=True):
        st.write('''
            - Data: Import data related to chemical trading.
            - :orange[**Gains/Losses**]: states with high inbound/outbound migration for selected year
            - :orange[**States Migration**]: percentage of states with annual inbound/outbound migration > 50,000
            ''')
        