from pathlib import Path
import streamlit as st
import pandas as pd
import tempfile
import os
from io import StringIO

# Add error handling for the import
try:
    from src.Engine import get_schema_engine
    SchemaMapEngine = get_schema_engine()
    METAHARMONIZER_AVAILABLE = True
except ImportError as e:
    METAHARMONIZER_AVAILABLE = False
    import_error = str(e)


def clear_session_state():
    """Clear mapping-related session state"""
    keys_to_clear = ['mapping_results', 'selected_columns', 'current_file']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    if 'upload_counter' in st.session_state:
        st.session_state['upload_counter'] += 1


def inject_custom_css():
    """Inject modern, minimalist custom CSS"""
    st.markdown("""
        <style>
        /* Import modern font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Global styles */
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Main container */
        .main {
            padding: 2rem 3rem;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        /* Headers */
        h1 {
            font-weight: 700;
            font-size: 2.5rem;
            color: #1a1a1a;
            margin-bottom: 0.5rem;
            letter-spacing: -0.02em;
        }
        
        h2 {
            font-weight: 600;
            font-size: 1.5rem;
            color: #2d2d2d;
            margin-top: 2.5rem;
            margin-bottom: 1rem;
            letter-spacing: -0.01em;
        }
        
        h3 {
            font-weight: 600;
            font-size: 1.1rem;
            color: #404040;
            margin-bottom: 0.75rem;
        }
        
        /* Remove emoji clutter, use clean text */
        .main h1::before,
        .main h2::before,
        .main h3::before {
            content: none;
        }
        
        /* Subtitle */
        .subtitle {
            font-size: 1.1rem;
            color: #666;
            font-weight: 400;
            margin-bottom: 3rem;
            line-height: 1.6;
        }
        
        /* Cards */
        .card {
            background: white;
            border: 1px solid #e5e5e5;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
            transition: box-shadow 0.2s ease;
        }
        
        .card:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }
        
        /* Buttons */
        .stButton > button {
            border-radius: 8px;
            font-weight: 500;
            border: none;
            padding: 0.6rem 1.5rem;
            transition: all 0.2s ease;
            letter-spacing: 0.01em;
        }
        
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
        }
        
        .stButton > button[kind="primary"]:hover {
            box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4);
            transform: translateY(-1px);
        }
        
        .stButton > button[kind="secondary"] {
            background: #f8f9fa;
            color: #495057;
            border: 1px solid #e5e5e5;
        }
        
        .stButton > button[kind="secondary"]:hover {
            background: #e9ecef;
            border-color: #dee2e6;
        }
        
        /* File uploader */
        .stFileUploader {
            border: 2px dashed #d0d0d0;
            border-radius: 12px;
            padding: 2rem;
            background: #fafafa;
            transition: all 0.2s ease;
        }
        
        .stFileUploader:hover {
            border-color: #667eea;
            background: #f8f9ff;
        }
        
        /* Radio buttons - cleaner style */
        .stRadio > div {
            gap: 1rem;
        }
        
        .stRadio > div > label {
            background: white;
            border: 1px solid #e5e5e5;
            border-radius: 8px;
            padding: 0.75rem 1.25rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .stRadio > div > label:hover {
            border-color: #667eea;
            background: #f8f9ff;
        }
        
        /* Multiselect */
        .stMultiSelect [data-baseweb="tag"] {
            background: #667eea;
            color: white;
            border-radius: 6px;
            font-size: 0.85rem;
            padding: 4px 8px;
            font-weight: 500;
        }
        
        .stMultiSelect [data-baseweb="select"] {
            border-radius: 8px;
            border-color: #e5e5e5;
        }
        
        /* Dataframe */
        .stDataFrame {
            border: 1px solid #e5e5e5;
            border-radius: 8px;
            overflow: hidden;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background: #fafafa;
            border-radius: 8px;
            font-weight: 500;
            color: #2d2d2d;
            border: 1px solid #e5e5e5;
        }
        
        .streamlit-expanderHeader:hover {
            background: #f0f0f0;
        }
        
        /* Info/Success/Warning boxes */
        .stAlert {
            border-radius: 8px;
            border: none;
            padding: 1rem 1.25rem;
            font-weight: 400;
        }
        
        /* Success message */
        .stSuccess {
            background: #d4edda;
            color: #155724;
        }
        
        /* Info message */
        .stInfo {
            background: #e7f3ff;
            color: #004085;
        }
        
        /* Warning message */
        .stWarning {
            background: #fff3cd;
            color: #856404;
        }
        
        /* Error message */
        .stError {
            background: #f8d7da;
            color: #721c24;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: #fafbfc;
            border-right: 1px solid #e5e5e5;
            padding: 2rem 1rem;
        }
        
        [data-testid="stSidebar"] h2 {
            font-size: 1.1rem;
            font-weight: 600;
            color: #2d2d2d;
            margin-bottom: 1.5rem;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            font-size: 1.8rem;
            font-weight: 600;
            color: #1a1a1a;
        }
        
        /* Spinner */
        .stSpinner > div {
            border-top-color: #667eea;
        }
        
        /* Download button special styling */
        .download-section {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 2rem;
            border: 1px solid #e5e5e5;
        }
        
        /* Stats grid */
        .stats-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }
        
        .stat-card {
            background: white;
            border: 1px solid #e5e5e5;
            border-radius: 8px;
            padding: 1.25rem;
            text-align: center;
        }
        
        .stat-value {
            font-size: 2rem;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 0.25rem;
        }
        
        .stat-label {
            font-size: 0.9rem;
            color: #666;
            font-weight: 500;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Reduce padding on mobile */
        @media (max-width: 768px) {
            .main {
                padding: 1rem;
            }
        }
        </style>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="MetaHarmonizer Schema Mapping",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inject custom CSS
    inject_custom_css()

    # Header
    st.markdown("<h1>MetaHarmonizer</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle'>Standardize clinical metadata schema for improved FAIRness</p>",
        unsafe_allow_html=True
    )

    # Check if MetaHarmonizer is available
    if not METAHARMONIZER_AVAILABLE:
        st.error(f"""
        **MetaHarmonizer Engine Not Found**
        
        {import_error}
        
        **Setup Instructions:**
        1. Clone the repository: `git clone https://github.com/shbrief/MetaHarmonizer`
        2. Install dependencies: `pip install -r requirements_sm.txt`
        3. Ensure the `src` folder is in your Python path
        """)
        return

    # Sidebar configuration
    with st.sidebar:
        st.markdown("## Configuration")
        
        mode = st.selectbox(
            "Mapping Mode",
            options=["auto", "manual"],
            help="Auto mode uses AI-powered automatic mapping"
        )

        top_k = st.slider(
            "Top Matches",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of mapping suggestions to display"
        )
        
        st.markdown("---")
        
        with st.expander("About"):
            st.markdown("""
            **MetaHarmonizer** harmonizes clinical data schemas across datasets.
            
            Built for researchers and data scientists working with heterogeneous clinical data.
            
            [Documentation](https://github.com/shbrief/MetaHarmonizer)
            """)

    # Data source selection
    st.markdown("## Data Source")
    
    data_source = st.radio(
        "Choose your data source",
        options=["Upload File", "Demo Dataset"],
        horizontal=True,
        label_visibility="collapsed"
    )

    if 'last_data_source' not in st.session_state or st.session_state['last_data_source'] != data_source:
        st.session_state['last_data_source'] = data_source
        clear_session_state()

    demo_file_path = "data/demo_data/clinical_metadata_demo.tsv"
    file_available = False
    selected_file = None
    all_columns = []

    if data_source == "Demo Dataset":
        if os.path.exists(demo_file_path):
            try:
                demo_df = pd.read_csv(demo_file_path, sep='\t')
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", f"{demo_df.shape[0]:,}")
                with col2:
                    st.metric("Columns", demo_df.shape[1])
                with col3:
                    st.metric("Source", "Demo")

                with st.expander("Preview Data"):
                    st.dataframe(demo_df.head(10), use_container_width=True)

                selected_file = demo_file_path
                file_available = True
                all_columns = demo_df.columns.tolist()
                
                if 'current_file' not in st.session_state or st.session_state['current_file'] != demo_file_path:
                    st.session_state['selected_columns'] = all_columns
                    st.session_state['current_file'] = demo_file_path
                    if 'mapping_results' in st.session_state:
                        del st.session_state['mapping_results']
                        
            except Exception as e:
                st.error(f"Unable to load demo file: {e}")
        else:
            st.error(f"Demo file not found at: `{demo_file_path}`")
    
    else:  # Upload File
        uploaded_file = st.file_uploader(
            "Drop your clinical data file here",
            type=['csv', 'tsv'],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            try:
                separator = '\t' if uploaded_file.name.endswith('.tsv') else ','
                df_preview = pd.read_csv(uploaded_file, sep=separator, nrows=10)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", f"{df_preview.shape[0]:,}+")
                with col2:
                    st.metric("Columns", df_preview.shape[1])
                with col3:
                    st.metric("Type", uploaded_file.name.split('.')[-1].upper())

                with st.expander("Preview Data"):
                    st.dataframe(df_preview, use_container_width=True)

                uploaded_file.seek(0)
                all_columns = df_preview.columns.tolist()
                
                if 'current_file' not in st.session_state or st.session_state['current_file'] != uploaded_file.name:
                    st.session_state['selected_columns'] = all_columns
                    st.session_state['current_file'] = uploaded_file.name
                    if 'mapping_results' in st.session_state:
                        del st.session_state['mapping_results']

                selected_file = uploaded_file
                file_available = True
                
            except Exception as e:
                st.error(f"Error reading file: {e}")

    # Column selection
    if file_available and len(all_columns) > 0:
        st.markdown("## Column Selection")
        
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("Select All", use_container_width=True):
                st.session_state['selected_columns'] = all_columns
                st.rerun()
        with col2:
            if st.button("Clear All", use_container_width=True):
                st.session_state['selected_columns'] = []
                st.rerun()

        if 'selected_columns' not in st.session_state:
            st.session_state['selected_columns'] = all_columns
        else:
            kept = [c for c in st.session_state['selected_columns'] if c in all_columns]
            if kept != st.session_state['selected_columns']:
                st.session_state['selected_columns'] = kept or all_columns

        selected_columns = st.multiselect(
            "Select columns to map",
            options=all_columns,
            default=st.session_state['selected_columns'],
            key="selected_columns",
            label_visibility="collapsed"
        )

        if len(selected_columns) == 0:
            st.warning("Please select at least one column")
            file_available = False
        else:
            st.caption(f"{len(selected_columns)} of {len(all_columns)} columns selected")

    # Processing section
    if file_available:
        st.markdown("## Schema Mapping")
        
        if st.button("Run Mapping", type="primary", use_container_width=False):
            with st.spinner("Processing..."):
                try:
                    # Handle file processing
                    if data_source == "Demo Dataset":
                        tmp_file_path = selected_file
                        cleanup_needed = False
                    else:
                        suffix = Path(selected_file.name).suffix.lower()
                        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=suffix) as tmp_file:
                            content = selected_file.read()
                            if isinstance(content, bytes):
                                content = content.decode('utf-8', errors='ignore')
                            tmp_file.write(content)
                            tmp_file_path = tmp_file.name
                        cleanup_needed = True

                    # Initialize engine and run mapping
                    engine = SchemaMapEngine(
                        clinical_data_path=tmp_file_path,
                        mode=mode,
                        top_k=top_k
                    )
                    results = engine.run_schema_mapping()

                    # Filter to selected columns
                    if 'selected_columns' in st.session_state and len(st.session_state['selected_columns']) > 0:
                        results = results[results['original_column'].isin(st.session_state['selected_columns'])]

                    # Remove _source columns
                    pattern = '_source'
                    columns_to_drop = results.columns[results.columns.str.contains(pattern, case=False)]
                    results.drop(columns=columns_to_drop, inplace=True)

                    # Store results
                    st.session_state['mapping_results'] = results
                    st.session_state['data_source'] = data_source
                    st.session_state['mode'] = mode
                    st.session_state['top_k'] = top_k

                    st.success("Mapping completed successfully")

                except Exception as e:
                    st.error(f"Mapping failed: {str(e)}")
                    with st.expander("Error Details"):
                        st.code(str(e))
                
                finally:
                    if cleanup_needed and 'tmp_file_path' in locals():
                        try:
                            os.unlink(tmp_file_path)
                        except:
                            pass

        # Display results
        if 'mapping_results' in st.session_state:
            results = st.session_state['mapping_results']

            st.markdown("---")
            st.markdown("## Results")

            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Mappings", len(results))
            with col2:
                st.metric("Unique Columns", results['original_column'].nunique())
            with col3:
                match1_col = 'match1_field'
                if match1_col in results.columns:
                    st.metric("Matched", results[match1_col].notna().sum())
            with col4:
                score_cols = [col for col in results.columns if col.endswith('_score')]
                if score_cols:
                    avg_score = results[score_cols[0]].mean()
                    if pd.notna(avg_score):
                        st.metric("Avg Score", f"{avg_score:.2f}")

            # View toggle
            col1, col2 = st.columns([4, 1])
            with col2:
                view_mode = st.selectbox(
                    "View",
                    options=["Simple", "Detailed"],
                    label_visibility="collapsed"
                )

            # Prepare display dataframe
            if view_mode == "Simple":
                simple_cols = ['original_column']
                for i in range(1, st.session_state.get('top_k', 5) + 1):
                    match_field_col = f'match{i}_field'
                    if match_field_col in results.columns:
                        simple_cols.append(match_field_col)
                display_df = results[simple_cols].copy()
                
                rename_dict = {'original_column': 'Original Column'}
                for i in range(1, st.session_state.get('top_k', 5) + 1):
                    match_field_col = f'match{i}_field'
                    if match_field_col in display_df.columns:
                        rename_dict[match_field_col] = f'Match {i}'
                display_df.rename(columns=rename_dict, inplace=True)
            else:
                display_df = results.copy()

            # Display dataframe
            st.dataframe(display_df, use_container_width=True, height=400)

            # Download section
            st.markdown("### Export Results")
            
            col1, col2 = st.columns(2)

            file_suffix = "demo" if st.session_state['data_source'] == "Demo Dataset" else "upload"
            
            with col1:
                csv_buffer_full = StringIO()
                results.to_csv(csv_buffer_full, index=False)
                st.download_button(
                    label="Download Full Results",
                    data=csv_buffer_full.getvalue(),
                    file_name=f"mapping_full_{file_suffix}_{st.session_state['mode']}_k{st.session_state['top_k']}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col2:
                simple_cols = ['original_column']
                for i in range(1, st.session_state.get('top_k', 5) + 1):
                    match_field_col = f'match{i}_field'
                    if match_field_col in results.columns:
                        simple_cols.append(match_field_col)
                simple_results = results[simple_cols].copy()
                csv_buffer_simple = StringIO()
                simple_results.to_csv(csv_buffer_simple, index=False)
                
                st.download_button(
                    label="Download Simple View",
                    data=csv_buffer_simple.getvalue(),
                    file_name=f"mapping_simple_{file_suffix}_{st.session_state['mode']}_k{st.session_state['top_k']}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            # Detailed statistics in expander
            with st.expander("Detailed Statistics"):
                score_cols = [col for col in results.columns if col.endswith('_score')]
                if score_cols:
                    st.markdown("**Average Match Scores**")
                    for score_col in score_cols:
                        avg_score = results[score_col].mean()
                        if pd.notna(avg_score):
                            match_num = score_col.replace('match', '').replace('_score', '')
                            st.write(f"Match {match_num}: {avg_score:.3f}")


if __name__ == "__main__":
    main()
