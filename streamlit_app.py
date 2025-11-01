import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# Set Streamlit page configuration
st.set_page_config(
    page_title="Project Samarth: Cross-Domain Q&A",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. Data Model (ETL & Normalization) ---

@st.cache_data
def load_and_normalize_data(crop_file_path, rainfall_file_path):
    """
    Loads, cleans, normalizes, and classifies the agricultural and climate data.
    This uses Streamlit's caching mechanism for fast reloading.
    """
    st.info(f"Starting data ingestion and normalization...")
    
    # Load Data
    try:
        crop_df = pd.read_csv(crop_file_path)
        rainfall_df = pd.read_csv(rainfall_file_path)
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None

    # --- Cleaning and Standardization ---
    crop_df['State_Name'] = crop_df['State_Name'].astype(str).str.lower().str.strip()
    crop_df['District_Name'] = crop_df['District_Name'].astype(str).str.lower().str.strip()
    crop_df['Crop'] = crop_df['Crop'].astype(str).str.lower().str.strip() 
    rainfall_df['SUBDIVISION'] = rainfall_df['SUBDIVISION'].astype(str).str.lower().str.strip()
    
    crop_df['Production'] = crop_df['Production'].fillna(0.0)
    crop_df['Area'] = crop_df['Area'].fillna(0.0)
    crop_df.rename(columns={'Crop_Year': 'YEAR'}, inplace=True)

    # 2. Geo-Spatial Mapping (IMD Sub-Division to State Mapping)
    subdivision_to_state_map = {
        # Northern India
        'jammu & kashmir': 'jammu and kashmir',
        'himachal pradesh': 'himachal pradesh',
        'uttarakhand': 'uttarakhand',
        'punjab': 'punjab',
        'haryana, chandigarh & delhi': 'haryana',
        'west uttar pradesh': 'uttar pradesh',
        'east uttar pradesh': 'uttar pradesh',

        'north bihar': 'bihar',
        'south bihar': 'bihar',
        'bihar plateau': 'bihar',

        # Western India
        'west rajasthan': 'rajasthan',
        'east rajasthan': 'rajasthan',
        'gujarat region': 'gujarat',
        'saurashtra & kutch': 'gujarat',

        # Central India
        'west madhya pradesh': 'madhya pradesh',
        'east madhya pradesh': 'madhya pradesh',
        'vidarbha': 'maharashtra',
        'madhya maharashtra': 'maharashtra',
        'marathwada': 'maharashtra',
        'konkan & goa': 'maharashtra',
        'chhattisgarh': 'chhattisgarh',

        # Eastern India
        'bihar plateau': 'bihar',
        'sub himalayan west bengal & sikkim': 'west bengal',
        'gangetic west bengal': 'west bengal',
        'jharkhand': 'jharkhand',
        'odisha': 'odisha',

        # Southern India
        'coastal andhra pradesh': 'andhra pradesh',
        'telangana': 'telangana',
        'rayalaseema': 'andhra pradesh',
        'tamil nadu': 'tamil nadu',
        'coastal karnataka': 'karnataka',
        'north interior karnataka': 'karnataka',
        'south interior karnataka': 'karnataka',
        'kerala': 'kerala',

        # Northeastern India
        'assam & meghalaya': 'assam',
        'nagaland-manipur-mizoram-tripura': 'manipur',
        'arunachal pradesh': 'arunachal pradesh',
    }

    
    rainfall_df['State_Name'] = rainfall_df['SUBDIVISION'].map(subdivision_to_state_map)
    missing_states = rainfall_df[rainfall_df['State_Name'].isna()]['SUBDIVISION'].unique()
    if len(missing_states) > 0:
        st.warning(f"⚠️ Missing state mappings for: {', '.join(missing_states[:5])}...")
    rainfall_analysis_df = rainfall_df.dropna(subset=['State_Name'])

    # 3. Aggregate Rainfall Data to State/Year level
    state_annual_rainfall_df = rainfall_analysis_df.groupby(['State_Name', 'YEAR'])['ANNUAL'].mean().reset_index()
    state_annual_rainfall_df.rename(columns={'ANNUAL': 'Avg_Annual_Rainfall_mm'}, inplace=True)

    # 4. Define Crop Classifications
    crop_class_map = {
        'rice': 'Cereal', 'wheat': 'Cereal', 'maize': 'Cereal', 'jowar': 'Cereal',
        'bajra': 'Cereal', 'ragi': 'Cereal', 'small millets': 'Cereal',
        'moong': 'Pulse', 'gram': 'Pulse', 'urad': 'Pulse', 'arhar/tur': 'Pulse', 'other kharif pulses': 'Pulse',
        'sugarcane': 'Water_Intensive', 'cotton(lint)': 'Water_Intensive',
        'bajra': 'Drought_Resistant', 'jowar': 'Drought_Resistant' 
    }
    
    crop_df['Crop_Type'] = crop_df['Crop'].map(crop_class_map)

    # Store data sources for traceability
    data_sources = {
        'crop_data': crop_file_path.name + " (MoA&FW)",
        'rainfall_data': rainfall_file_path.name + " (IMD)"
    }
    
    final_rainfall_df = state_annual_rainfall_df[['YEAR', 'State_Name', 'Avg_Annual_Rainfall_mm']]
    st.success("Data ingestion and harmonization successful!")

    return crop_df, final_rainfall_df, data_sources

# --- 2. Intelligent Q&A and Reasoning Engine Class ---

class SamarthQAEngine:
    def __init__(self, crop_df, rainfall_df, sources):
        self.crop_df = crop_df
        self.rainfall_df = rainfall_df
        self.sources = sources
        self.min_year = self.crop_df['YEAR'].min()
        self.max_year = self.crop_df['YEAR'].max()
        self.max_rainfall_year = self.rainfall_df['YEAR'].max()
        self.available_states = sorted(self.crop_df['State_Name'].unique())

    def _get_time_period(self, n_years):
        """Calculates the max common time window available in both datasets."""
        end_year = min(self.max_year, self.max_rainfall_year)
        start_year = end_year - n_years + 1
        actual_n = end_year - start_year + 1
        return int(start_year), int(end_year), int(actual_n)

    def _get_citation(self, start_year, end_year, actual_n):
         return f"""
**Data Sources:**
1. Agricultural Data: {self.sources['crop_data']}
2. Climate Data: {self.sources['rainfall_data']}

**Analysis Period:** {start_year} - {end_year} ({actual_n} Years)
"""
    # ----------------------------------------------------------------------
    # QUERY 1: Compare rainfall and top M crops
    # ----------------------------------------------------------------------
    def query_1_compare_climate_crops(self, state_x, state_y, n_years, crop_type_c='Cereal', m=3):
        
        start_year, end_year, actual_n = self._get_time_period(n_years)
        state_x, state_y = state_x.lower(), state_y.lower()
        target_states = [state_x, state_y]
        results = []

        for state in target_states:
            rainfall_avg = round(
                self.rainfall_df[
                    (self.rainfall_df['State_Name'] == state) &
                    (self.rainfall_df['YEAR'].between(start_year, end_year))
                ]['Avg_Annual_Rainfall_mm'].mean(), 2
            )
            
            crop_subset = self.crop_df[
                (self.crop_df['State_Name'] == state) &
                (self.crop_df['YEAR'].between(start_year, end_year)) &
                (self.crop_df['Crop_Type'] == crop_type_c)
            ]
            
            crop_totals = crop_subset.groupby('Crop')['Production'].sum().reset_index()
            top_crops = crop_totals.nlargest(m, 'Production')
            top_crops['Production_Lakh_Tonnes'] = (top_crops['Production'] / 100000).round(2)
            
            crop_summary = [f"{row['Crop'].title()} ({row['Production_Lakh_Tonnes']} LT)" for _, row in top_crops.iterrows()]
            
            results.append({
                'State': state.title(),
                'Avg Annual Rainfall (mm)': rainfall_avg if not pd.isna(rainfall_avg) else 'N/A',
                f'Top {m} {crop_type_c} Crops': '; '.join(crop_summary)
            })

        final_df = pd.DataFrame(results)
        
        return "### Query 1 Result: Compare Climate and Top Crops\n" + final_df.to_markdown(index=False, numalign="left", stralign="left") + "\n" + self._get_citation(start_year, end_year, actual_n)

    # ----------------------------------------------------------------------
    # QUERY 2: Identify district with highest/lowest production
    # ----------------------------------------------------------------------
    def query_2_district_comparison(self, state_x, state_y, crop_z):
        
        # Find the latest year with data for this crop
        available_years = self.crop_df[self.crop_df['Crop'] == crop_z]['YEAR'].dropna().unique()
        if len(available_years) == 0:
            st.error(f"No data found for crop '{crop_z.title()}' in any year.")
            return "No valid data found for this crop."

        latest_year = int(max(available_years))
        filtered_crops = self.crop_df[
            (self.crop_df['Crop'] == crop_z) & 
            (self.crop_df['YEAR'] == latest_year)
        ].copy()

        
        filtered_crops = self.crop_df[
            (self.crop_df['Crop'] == crop_z) &
            (self.crop_df['YEAR'] == latest_year)
        ].copy()

        results = []
        
        def find_extreme_district(data, extreme='max'):
            if data.empty: return None
            if extreme == 'min':
                 data_non_zero = data[data['Production'] > 0]
                 if data_non_zero.empty: data_non_zero = data
                 idx = data_non_zero['Production'].idxmin() 
            else:
                 idx = data['Production'].idxmax()
            return data.loc[idx]

        prod_x = filtered_crops[filtered_crops['State_Name'] == state_x]
        max_district = find_extreme_district(prod_x, 'max')
        
        prod_y = filtered_crops[filtered_crops['State_Name'] == state_y]
        min_district = find_extreme_district(prod_y, 'min')
        
        
        if max_district is not None and max_district['Production'] > 0:
            results.append({
                'Metric': f'Highest Production of {crop_z.title()}',
                'State': max_district['State_Name'].title(),
                'District': max_district['District_Name'].title(),
                'Production (Tonnes)': f"{max_district['Production']:,.0f}"
            })
        else:
            results.append({
                'Metric': f'Highest Production of {crop_z.title()}',
                'State': state_x.title(),
                'District': 'N/A',
                'Production (Tonnes)': f'N/A (No data found in {latest_year})'
            })

        if min_district is not None:
            results.append({
                'Metric': f'Lowest Production of {crop_z.title()}',
                'State': min_district['State_Name'].title(),
                'District': min_district['District_Name'].title(),
                'Production (Tonnes)': f"{min_district['Production']:,.0f}"
            })
        else:
            results.append({'Metric': f'Lowest Production of {crop_z.title()}', 'State': state_y.title(), 'District': 'N/A', 'Production (Tonnes)': f'N/A (No data found in {latest_year})'})
            
        final_df = pd.DataFrame(results)
        
        citation = f"""
**Data Source:**
1. Agricultural Data: {self.sources['crop_data']}

**Analysis Year:** {latest_year} (based on available crop data)

"""
        return "### Query 2 Result: District Production Comparison\n" + final_df.to_markdown(index=False, numalign="left", stralign="left") + "\n" + citation

    # ----------------------------------------------------------------------
    # QUERY 3: Analyze production trend and correlate with climate
    # ----------------------------------------------------------------------
    def query_3_trend_correlation(self, state, crop_type_c, n_years):
        
        state = state.lower()
        start_year, end_year, actual_n = self._get_time_period(n_years)

        prod_trend_df = self.crop_df[
            (self.crop_df['State_Name'] == state) &
            (self.crop_df['Crop_Type'] == crop_type_c) &
            (self.crop_df['YEAR'].between(start_year, end_year))
        ].groupby(['State_Name', 'YEAR'])['Production'].sum().reset_index()

        rain_trend_df = self.rainfall_df[
            (self.rainfall_df['State_Name'] == state) &
            (self.rainfall_df['YEAR'].between(start_year, end_year))
        ].copy()

        trend_df = pd.merge(prod_trend_df, rain_trend_df, on=['YEAR', 'State_Name'], how='inner')
        
        correlation = 'Insufficient Data'
        
        if len(trend_df) > 1:
            prod_mean = trend_df['Production'].mean()
            prod_std = trend_df['Production'].std()
            trend_df = trend_df[trend_df['Production'] > 0]
            trend_df = trend_df[abs(trend_df['Production'] - prod_mean) < 3 * prod_std]
            
            if len(trend_df) > 1:
                 r, p_value = pearsonr(trend_df['Production'], trend_df['Avg_Annual_Rainfall_mm'])
                 correlation = round(r, 3)
            else:
                 correlation = 'Insufficient Data after Outlier Removal'
        
        prod_trend = "stable"
        if not trend_df.empty and len(trend_df) > 1:
            prod_delta = trend_df['Production'].iloc[-1] - trend_df['Production'].iloc[0]
            if prod_delta > 0: prod_trend = "increasing"
            elif prod_delta < 0: prod_trend = "decreasing"
        
        # Synthesize Impact Summary 
        impact = "Cannot establish a reliable correlation or trend due to data limitations or sparsity."
        if isinstance(correlation, float):
            abs_corr = abs(correlation)
            if abs_corr > 0.7: strength = "very strong"
            elif abs_corr > 0.5: strength = "strong"
            elif abs_corr > 0.3: strength = "moderate"
            else: strength = "weak"

            if correlation > 0:
                impact_type = "positive"
                conclusion = "production tends to rise with annual rainfall."
            else:
                impact_type = "negative"
                conclusion = "production tends to fall when rainfall is higher (e.g., due to flooding/excess moisture)."
                
            impact = f"A **{strength} {impact_type} impact** (Correlation: ${correlation}$), suggesting {conclusion}"

        # Prepare final table for trend visualization
        trend_table = trend_df[['YEAR', 'Production', 'Avg_Annual_Rainfall_mm']].copy()
        trend_table.rename(columns={'Production': 'Total Production (Tonnes)', 'Avg_Annual_Rainfall_mm': 'Rainfall (mm)'}, inplace=True)
        trend_table['Total Production (Tonnes)'] = trend_table['Total Production (Tonnes)'].apply(lambda x: f"{x:,.0f}")
        
        summary = f"""
**Summary of Apparent Impact (LLM Synthesis):**  

The overall production trend for **{crop_type_c}** crops in **{state.title()}** was **{prod_trend}**.  

Correlation between Production and Average Annual Rainfall (Pearson $\\rho$): **{correlation}**.  

**Apparent Impact:** {impact}
"""
        return "### Query 3 Result: Production Trend & Climate Correlation\n" + summary + "\n**Time Series Data:**\n" + trend_table.to_markdown(index=False, numalign="left", stralign="left") + "\n" + self._get_citation(start_year, end_year, actual_n)

    # ----------------------------------------------------------------------
    # QUERY 4: Policy arguments (Synthesizing two data sources)
    # ----------------------------------------------------------------------
    def query_4_policy_arguments(self, state, crop_a_type, crop_b_type, n_years):
        
        state = state.lower()
        start_year, end_year, actual_n = self._get_time_period(n_years)
        
        policy_df = self.crop_df[
            (self.crop_df['State_Name'] == state) &
            (self.crop_df['YEAR'].between(start_year, end_year)) &
            (self.crop_df['Crop_Type'].isin([crop_a_type, crop_b_type]))
        ]
        
        analysis_df = pd.merge(
            policy_df,
            self.rainfall_df,
            on=['YEAR', 'State_Name'],
            how='inner'
        )

        metrics = analysis_df.groupby('Crop_Type').agg(
            Total_Production=('Production', 'sum'),
            Total_Area=('Area', 'sum')
        )
        
        rainfall_values = analysis_df['Avg_Annual_Rainfall_mm'].dropna()
        avg_rainfall = round(rainfall_values.mean(), 2) if not rainfall_values.empty else 0
        
        arguments = []
        
        if crop_a_type in metrics.index and crop_b_type in metrics.index and analysis_df['YEAR'].nunique() > 1:
            try:
                prod_a = metrics.loc[crop_a_type, 'Total_Production'] / 100000
                prod_b = metrics.loc[crop_b_type, 'Total_Production'] / 100000
                area_a = metrics.loc[crop_a_type, 'Total_Area']
                area_b = metrics.loc[crop_b_type, 'Total_Area']
                
                # Argument 1: Yield Stability
                yield_a = (prod_a * 100000) / area_a if area_a > 0 else 0
                yield_b = (prod_b * 100000) / area_b if area_b > 0 else 0
                
                arg1 = (
                    f"**Yield per Hectare (Efficiency):** Over the {actual_n} year period, {crop_a_type.title()} crops yield "
                    f"${yield_a:.1f}$ Tonnes/Hectare (T/Ha), while {crop_b_type.title()} yields ${yield_b:.1f}$ T/Ha. "
                    f"Promoting the drought-resistant crop ensures **more stable production** when water is scarce, fulfilling the core policy objective of risk mitigation."
                )
                arguments.append(arg1)
                
                # Argument 2: Climate Correlation (Water Risk)
                yearly_analysis = analysis_df.groupby(['YEAR', 'Crop_Type', 'State_Name']).agg(
                    Yearly_Production=('Production', 'sum'),
                    Avg_Annual_Rainfall_mm=('Avg_Annual_Rainfall_mm', 'mean')
                ).reset_index()

                corr_a = yearly_analysis[yearly_analysis['Crop_Type'] == crop_a_type]['Yearly_Production'].corr(yearly_analysis['Avg_Annual_Rainfall_mm'])
                corr_b = yearly_analysis[yearly_analysis['Crop_Type'] == crop_b_type]['Yearly_Production'].corr(yearly_analysis['Avg_Annual_Rainfall_mm'])
                
                if not pd.isna(corr_a) and not pd.isna(corr_b):
                    arg2 = (
                        f"**Climate Risk Mitigation:** The production of {crop_a_type.title()} has a correlation of "
                        f"$\\rho = {corr_a:.3f}$ with annual rainfall, significantly lower than {crop_b_type.title()}'s correlation of $\\rho = {corr_b:.3f}$. "
                        f"The weaker climate dependency for {crop_a_type.title()} means it is a **safer investment** during fluctuating monsoon years."
                    )
                else:
                    arg2 = "Climate correlation data is insufficient to form an argument."
                arguments.append(arg2)
                
                # Argument 3: Scale/Water Footprint 
                arg3 = (
                    f"**Water Footprint & Scaling:** Given the state's average annual rainfall of **{avg_rainfall:,.2f} mm**, "
                    f"promoting {crop_a_type.title()} saves water. The water-intensive crop, {crop_b_type.title()}, occupied **{area_b:,.0f} hectares** of land. "
                    f"A policy shift frees up water resources from this large area, improving the state's **overall water security**."
                )
                arguments.append(arg3)

            except Exception as e:
                arguments = [f"An internal calculation error occurred while generating metrics: {e}"] * 3
        else:
            arguments = ["Data for one or both crop types is missing or sparse in the available period. Cannot generate data-backed arguments."] * 3


        argument_summary = "\n".join([f"{i+1}. {arg}" for i, arg in enumerate(arguments)])
        
        return f"""
### Query 4 Result: Data-Backed Policy Arguments
**Policy:** Promote {crop_a_type.title()} (e.g., Bajra) over {crop_b_type.title()} (e.g., Sugarcane) in {state.title()}.
**State Average Annual Rainfall:** {avg_rainfall:,.2f} mm

**Arguments based on historical data ({start_year}-{end_year}):**
{argument_summary}

{self._get_citation(start_year, end_year, actual_n)}
"""


# --- 3. Streamlit Application Interface ---

def main():
    st.title("Project Samarth: Intelligent Cross-Domain Q&A Prototype")
    st.markdown("---")
    
    # 1. File Upload Section
    st.sidebar.header("1. Upload Data Files")
    # Streamlit uses file_uploader to get the file object
    crop_file = st.sidebar.file_uploader("Upload Crop Production File (crop.csv)", type=["csv"])
    rainfall_file = st.sidebar.file_uploader("Upload Rainfall Analysis File (Rainfall_State_Analysis_India_1901_2017.csv)", type=["csv"])
    
    qa_system = None
    CROP_DF = None
    RAINFALL_DF = None
    available_policy_types = [] # Initialize outside the if block

    if crop_file and rainfall_file:
        # Load and normalize the data
        CROP_DF, RAINFALL_DF, DATA_SOURCES = load_and_normalize_data(crop_file, rainfall_file)
        
        # Check explicitly that the dataframes were created before proceeding
        if CROP_DF is not None and RAINFALL_DF is not None:
            qa_system = SamarthQAEngine(CROP_DF, RAINFALL_DF, DATA_SOURCES)
            st.sidebar.success("Data successfully loaded and normalized!")
            
            # Display stats in sidebar
            st.sidebar.markdown(f"**Data Summary**")
            st.sidebar.markdown(f"- **Crop Years:** {qa_system.min_year} to {qa_system.max_year}")
            st.sidebar.markdown(f"- **Rainfall Years:** {qa_system.rainfall_df['YEAR'].min()} to {qa_system.max_rainfall_year}")
            st.sidebar.markdown(f"- **Available States:** {len(qa_system.available_states)}")

            # Dynamically get available crop types for policy analysis
            available_policy_types = sorted(CROP_DF['Crop_Type'].dropna().unique())
        else:
            st.sidebar.error("Failed to process data. Check file contents.")
            # Do not return here; allow the rest of the app to render with warnings
    else:
        st.sidebar.warning("Please upload both `crop.csv` and `Rainfall_State_Analysis_India_1901_2017.csv` to proceed.")
        
    # --- Conditional Rendering of the Q&A Interface ---
    
    if qa_system is None:
        st.info("The Q&A interface will become active once both data files are successfully uploaded and processed in the sidebar.")
        return # Stop execution if data isn't ready

    st.header("2. Run Cross-Domain Queries")
    query_option = st.selectbox(
        "Select Query Type:",
        ["Query 1: Compare Climate & Top Crops", 
         "Query 2: Highest/Lowest District Production", 
         "Query 3: Trend & Correlation Analysis",
         "Query 4: Policy Argument Synthesis"]
    )
    
    st.markdown("---")
    
    # --- QUERY 1: Compare Climate & Top Crops ---
    if query_option == "Query 1: Compare Climate & Top Crops":
        st.subheader("Query 1: Compare Rainfall and Top M Crops of Type C")
        col1, col2, col3 = st.columns(3)
        
        # FIX: Ensure index calculation doesn't fail if state list is small
        try:
            default_index_maharashtra = qa_system.available_states.index('maharashtra')
        except ValueError:
            default_index_maharashtra = 0
        try:
            default_index_gujarat = qa_system.available_states.index('gujarat')
        except ValueError:
            default_index_gujarat = 0

        state_x = col1.selectbox("Select State X:", qa_system.available_states, index=default_index_maharashtra)
        state_y = col2.selectbox("Select State Y:", qa_system.available_states, index=default_index_gujarat)
        
        n_years = col1.slider("Analysis Period (N Years):", min_value=3, max_value=20, value=10)
        m_crops = col2.slider("Top M Crops:", min_value=1, max_value=5, value=3)
        crop_type = col3.selectbox("Crop Type (C):", ['Cereal', 'Pulse', 'Water_Intensive'])
        
        if st.button(f"Run Query 1 for {state_x.title()} vs {state_y.title()}"):
            with st.spinner("Executing KG Query and Synthesizing Results..."):
                result = qa_system.query_1_compare_climate_crops(state_x, state_y, n_years, crop_type, m_crops)
                st.markdown(result)
    
    # --- QUERY 2: Highest/Lowest District Production ---
    elif query_option == "Query 2: Highest/Lowest District Production":
        st.subheader("Query 2: Identify District Extremes for Crop Z")
        col1, col2 = st.columns(2)

        try:
            default_index_maharashtra = qa_system.available_states.index('maharashtra')
        except ValueError:
            default_index_maharashtra = 0
        try:
            default_index_gujarat = qa_system.available_states.index('gujarat')
        except ValueError:
            default_index_gujarat = 0
        
        state_x = col1.selectbox("State X (Highest District):", qa_system.available_states, index=default_index_maharashtra, key='q2_x')
        state_y = col2.selectbox("State Y (Lowest District):", qa_system.available_states, index=default_index_gujarat, key='q2_y')
        
        all_crops = sorted(qa_system.crop_df['Crop'].unique())
        try:
            default_index_rice = all_crops.index('rice')
        except ValueError:
            default_index_rice = 0
            
        crop_z = st.selectbox("Select Crop Z:", all_crops, index=default_index_rice)
        
        if st.button(f"Run Query 2 for {crop_z.title()}"):
            with st.spinner(f"Querying production in year {qa_system.max_year}..."):
                result = qa_system.query_2_district_comparison(state_x, state_y, crop_z)
                st.markdown(result)
    
    # --- QUERY 3: Trend & Correlation Analysis ---
    elif query_option == "Query 3: Trend & Correlation Analysis":
        st.subheader("Query 3: Correlate Production Trend with Climate")
        col1, col2 = st.columns(2)

        try:
            default_index_maharashtra = qa_system.available_states.index('maharashtra')
        except ValueError:
            default_index_maharashtra = 0
        
        state = col1.selectbox("Select State:", qa_system.available_states, index=default_index_maharashtra, key='q3_state')
        crop_type = col2.selectbox("Crop Type:", ['Cereal', 'Pulse', 'Water_Intensive'], key='q3_crop')
        n_years = st.slider("Analysis Period (N Years):", min_value=5, max_value=20, value=10, key='q3_years')
        
        if st.button(f"Run Query 3 for {state.title()}"):
            with st.spinner("Calculating Trend and Pearson Correlation..."):
                result = qa_system.query_3_trend_correlation(state, crop_type, n_years)
                st.markdown(result)

    # --- QUERY 4: Policy Argument Synthesis (Updated to use dynamic crop types) ---
    elif query_option == "Query 4: Policy Argument Synthesis":
        st.subheader("Query 4: Data-Backed Policy Arguments")
        st.markdown("_Policy: Promote Crop A (e.g., Drought-Resistant) over Crop B (e.g., Water-Intensive)._")
        
        col1, col2, col3 = st.columns(3)

        try:
            default_index_gujarat = qa_system.available_states.index('gujarat')
        except ValueError:
            default_index_gujarat = 0
        
        state = col1.selectbox("Select Target State:", qa_system.available_states, index=default_index_gujarat, key='q4_state')
        
        # Dynamically populate crop types based on available data
        if not available_policy_types:
            st.error("No valid crop types found in the data for policy analysis (check 'Crop_Type' column).")
            return
            
        default_index_a = available_policy_types.index('Drought_Resistant') if 'Drought_Resistant' in available_policy_types else 0
        default_index_b = available_policy_types.index('Water_Intensive') if 'Water_Intensive' in available_policy_types and len(available_policy_types) > 1 else (0 if len(available_policy_types) == 1 else 1)

        crop_a_type = col2.selectbox("Promoted Crop Type (A):", available_policy_types, index=default_index_a, help="e.g., Drought_Resistant")
        crop_b_type = col3.selectbox("Replaced Crop Type (B):", available_policy_types, index=default_index_b, help="e.g., Water_Intensive")
        
        n_years = st.slider("Historical Data Period (N Years):", min_value=5, max_value=20, value=10, key='q4_years')
        
        if st.button(f"Generate 3 Policy Arguments for {state.title()}"):
            with st.spinner("Synthesizing multi-factor arguments..."):
                if crop_a_type == crop_b_type:
                    st.warning("Please select two different crop types for the promotion/replacement policy.")
                else:
                    result = qa_system.query_4_policy_arguments(state, crop_a_type, crop_b_type, n_years)
                    st.markdown(result)


# --- RUN THE APP ---
if __name__ == "__main__":
    main()

