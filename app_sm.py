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


def main():
    st.set_page_config(page_title="MetaHarmonizer Schema Mapping",
                       page_icon="🔗",
                       layout="wide")

    st.title("🔗 MetaHarmonizer Schema Mapping Tool")
    st.markdown(
        "Upload your clinical data file and run schema mapping using MetaHarmonizer"
    )

    # Check if MetaHarmonizer is available
    if not METAHARMONIZER_AVAILABLE:
        st.error(f"""
        ❌ **MetaHarmonizer not found!**
        
        Error: {import_error}
        
        **Setup Instructions:**
        1. Clone the repository: `git clone https://github.com/shbrief/MetaHarmonizer`
        2. Install dependencies: `pip install -r requirements_sm.txt`
        3. Make sure the `src` folder is in your Python path
        
        Or run this app from within the MetaHarmonizer directory.
        """)
        return

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    mode = st.sidebar.selectbox(
        "Mapping Mode",
        options=["manual", "auto"],
        help="Choose between manual or automatic schema mapping")

    top_k = st.sidebar.slider("Top K Matches",
                              min_value=1,
                              max_value=20,
                              value=5,
                              help="Number of top matches to return")

    # File selection section
    st.header("📁 Choose Your Data Source")

    # Radio button to choose between demo or upload
    data_source = st.radio(
        "Select data source:",
        options=["Upload your own file", "Use demo file"],
        help="Choose whether to upload your own data or test with the demo file"
    )

    if 'last_data_source' not in st.session_state or st.session_state[
            'last_data_source'] != data_source:
        st.session_state['last_data_source'] = data_source
        clear_session_state()

    demo_file_path = "data/demo_data/clinical_metadata_demo.tsv"

    if data_source == "Use demo file":
        # Demo file section
        st.subheader("🎯 Demo File Selected")

        if os.path.exists(demo_file_path):
            try:
                demo_df = pd.read_csv(demo_file_path, sep='\t')
                st.success(f"✅ Demo file loaded: `{demo_file_path}`")
                st.write(
                    f"**Shape:** {demo_df.shape[0]} rows × {demo_df.shape[1]} columns"
                )

                with st.expander("Preview demo data"):
                    st.dataframe(demo_df.head(10))
                    if len(demo_df) > 10:
                        st.info(
                            f"Showing first 10 rows of {len(demo_df)} total rows"
                        )

                # Set variables for processing
                selected_file = demo_file_path
                file_available = True
                all_columns = demo_df.columns.tolist()
                if 'current_file' not in st.session_state or st.session_state[
                        'current_file'] != demo_file_path:
                    st.session_state['selected_columns'] = all_columns
                    st.session_state['current_file'] = demo_file_path
                    if 'mapping_results' in st.session_state:
                        del st.session_state['mapping_results']
            except Exception as e:
                st.error(f"Could not load demo file: {e}")
                file_available = False
                selected_file = None
                all_columns = []
                clear_session_state()
        else:
            st.error(f"Demo file not found at: `{demo_file_path}`")
            file_available = False
            selected_file = None
            all_columns = []
            clear_session_state()
    else:
        # File upload section
        st.subheader("📤 Upload Your File")

        uploaded_file = st.file_uploader(
            "Choose a clinical data file",
            type=['csv', 'tsv'],
            help="Upload a CSV or TSV file containing clinical data")

        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")

            # Show file preview
            with st.expander("👀 Preview uploaded data"):
                try:
                    # Determine separator based on file extension
                    if uploaded_file.name.endswith('.tsv'):
                        separator = '\t'
                    else:
                        separator = ','

                    # Read and display preview
                    df_preview = pd.read_csv(uploaded_file,
                                             sep=separator,
                                             nrows=5)
                    st.dataframe(df_preview)
                    st.info(
                        f"File shape: {df_preview.shape[0]}+ rows × {df_preview.shape[1]} columns"
                    )

                    # Reset file pointer for processing
                    uploaded_file.seek(0)

                    # Get all columns
                    all_columns = df_preview.columns.tolist()
                    if 'current_file' not in st.session_state or st.session_state[
                            'current_file'] != uploaded_file.name:
                        st.session_state['selected_columns'] = all_columns
                        st.session_state['current_file'] = uploaded_file.name
                        if 'mapping_results' in st.session_state:
                            del st.session_state['mapping_results']
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                    file_available = False
                    selected_file = None
                    all_columns = []
                    return

            # Set variables for processing
            selected_file = uploaded_file
            file_available = True
        else:
            file_available = False
            selected_file = None
            all_columns = []
            clear_session_state()

    # Column selection section (only show if file is available)
    if file_available and len(all_columns) > 0:
        st.header("📋 Select Columns for Mapping")

        with st.expander("Click to select columns", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Select All", key="select_all"):
                    st.session_state['selected_columns'] = all_columns
                    st.rerun()
            with col2:
                if st.button("Deselect All", key="deselect_all"):
                    st.session_state['selected_columns'] = []
                    st.rerun()

            st.markdown("""
                <style>
                .stMultiSelect [data-baseweb="tag"] {
                    font-size: 0.85rem;
                    height: auto;
                    padding: 2px 4px;
                    color: black;
                    background-color: lightgrey;
                }
                .stMultiSelect [data-baseweb="select"] {
                    font-size: 0.85rem;
                    margin-top: -1.5rem;
                }
                button[kind="secondary"] {
                    font-size: 0.85rem !important;
                    padding: 0.1rem 0.4rem !important;
                    height: auto !important;
                    min-height: 1rem !important;
                }
                button[kind="secondary"] > div > p {
                    font-size: 0.85rem !important;
                }
                </style>
            """,
                        unsafe_allow_html=True)

            if 'selected_columns' not in st.session_state:
                st.session_state['selected_columns'] = all_columns
            else:
                kept = [
                    c for c in st.session_state['selected_columns']
                    if c in all_columns
                ]
                if kept != st.session_state['selected_columns']:
                    st.session_state['selected_columns'] = kept or all_columns

            selected_columns = st.multiselect(
                "Choose columns to map:",
                options=all_columns,
                help=
                "Select which columns you want to perform schema mapping on",
                label_visibility="collapsed",
                key="selected_columns",
            )

        if len(selected_columns) == 0:
            st.warning("⚠️ Please select at least one column to continue")
            file_available = False
        else:
            st.success(
                f"✅ {len(selected_columns)} column(s) selected for mapping")

    # Processing section (only show if file is available)
    if file_available:
        # Processing section
        st.header("🔄 Schema Mapping")

        if st.button("🚀 Run Schema Mapping", type="primary"):
            with st.spinner(
                    "Processing schema mapping... This may take a few minutes."
            ):
                info_ph = st.empty()  # Placeholder for info messages
                try:
                    # Handle file processing based on source
                    if data_source == "Use demo file":
                        # Use demo file directly
                        tmp_file_path = selected_file
                        cleanup_needed = False
                        info_ph.info("Processing demo file...")
                    else:
                        # Create temporary file for uploaded file, preserve original suffix
                        suffix = Path(selected_file.name).suffix.lower()

                        with tempfile.NamedTemporaryFile(
                                mode='w+', delete=False,
                                suffix=suffix) as tmp_file:
                            content = selected_file.read()
                            if isinstance(content, bytes):
                                content = content.decode('utf-8',
                                                         errors='ignore')
                            tmp_file.write(content)
                            tmp_file_path = tmp_file.name

                        cleanup_needed = True
                        info_ph.info("Processing uploaded file...")

                    # Initialize the SchemaMapEngine
                    engine = SchemaMapEngine(clinical_data_path=tmp_file_path,
                                             mode=mode,
                                             top_k=top_k)

                    # Run schema mapping
                    results = engine.run_schema_mapping()

                    # Filter results to only include selected columns
                    if 'selected_columns' in st.session_state and len(
                            st.session_state['selected_columns']) > 0:
                        results = results[results['original_column'].isin(
                            st.session_state['selected_columns'])]
                        info_ph.info(
                            f"Filtered to {len(results)} selected columns...")

                    # Remove `_source` columns
                    pattern = '_source'
                    columns_to_drop = results.columns[
                        results.columns.str.contains(pattern, case=False)]
                    results.drop(columns=columns_to_drop, inplace=True)

                    # Store results in session state for persistent display
                    st.session_state['mapping_results'] = results
                    st.session_state['data_source'] = data_source
                    st.session_state['mode'] = mode
                    st.session_state['top_k'] = top_k

                    # Clean up temporary file if needed
                    if cleanup_needed:
                        os.unlink(tmp_file_path)

                    st.success("✅ Schema mapping completed successfully!")

                except Exception as e:
                    st.error(f"❌ Error during schema mapping: {str(e)}")

                    # More detailed error info in expander
                    with st.expander("🔍 Detailed Error Information"):
                        st.code(str(e))
                        st.write("**Possible solutions:**")
                        st.write("- Check if your file format is correct")
                        st.write(
                            "- Ensure the file contains valid clinical data")
                        st.write("- Try a different mode (manual/auto)")
                        st.write(
                            "- Check if all MetaHarmonizer dependencies are installed"
                        )

                finally:
                    info_ph.empty()

        # Display results if available in session state
        if 'mapping_results' in st.session_state:
            results = st.session_state['mapping_results']

            # Results section
            st.header("📊 Results")

            # Add toggle for view mode
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader("Mapping Results")
            with col2:
                view_mode = st.selectbox(
                    "View Mode",
                    options=["Detailed View", "Simple View"],
                    help=
                    "Toggle between detailed results and simplified match view"
                )

            # Prepare dataframe based on view mode
            if view_mode == "Simple View":
                # Simple view: only original_column and match fields
                simple_cols = ['original_column']

                # Add match columns dynamically based on what's available
                for i in range(1, st.session_state.get('top_k', 5) + 1):
                    match_field_col = f'match{i}_field'
                    if match_field_col in results.columns:
                        simple_cols.append(match_field_col)

                display_df = results[simple_cols].copy()

                # Rename columns for better readability
                rename_dict = {'original_column': 'Original Column'}
                for i in range(1, st.session_state.get('top_k', 5) + 1):
                    match_field_col = f'match{i}_field'
                    if match_field_col in display_df.columns:
                        rename_dict[match_field_col] = f'Match {i}'

                display_df.rename(columns=rename_dict, inplace=True)

            else:
                # Detailed view: all columns
                display_df = results.copy()

            # Display the dataframe with limited height to prevent scrolling issues
            st.dataframe(display_df, use_container_width=True, height=400)

            # Download buttons
            st.subheader("📥 Download Results")

            col1, col2 = st.columns(2)

            with col1:
                # Download full results
                csv_buffer_full = StringIO()
                results.to_csv(csv_buffer_full, index=False)

                file_suffix = "demo" if st.session_state[
                    'data_source'] == "Use demo file" else "uploaded"
                st.download_button(
                    label="Download Full Results (CSV)",
                    data=csv_buffer_full.getvalue(),
                    file_name=
                    f"schema_mapping_full_{file_suffix}_{st.session_state['mode']}_top{st.session_state['top_k']}.csv",
                    mime="text/csv")

            with col2:
                # Download simple view
                simple_cols = ['original_column']
                for i in range(1, st.session_state.get('top_k', 5) + 1):
                    match_field_col = f'match{i}_field'
                    if match_field_col in results.columns:
                        simple_cols.append(match_field_col)

                simple_results = results[simple_cols].copy()
                csv_buffer_simple = StringIO()
                simple_results.to_csv(csv_buffer_simple, index=False)

                st.download_button(
                    label="Download Simple View (CSV)",
                    data=csv_buffer_simple.getvalue(),
                    file_name=
                    f"schema_mapping_simple_{file_suffix}_{st.session_state['mode']}_top{st.session_state['top_k']}.csv",
                    mime="text/csv")

            # Summary statistics
            with st.expander("Summary Statistics"):
                st.write(f"**Total mappings found:** {len(results)}")
                st.write(
                    f"**Unique original columns:** {results['original_column'].nunique()}"
                )

                # Count columns with matches
                match1_col = 'match1_field'
                if match1_col in results.columns:
                    non_null_matches = results[match1_col].notna().sum()
                    st.write(
                        f"**Columns with at least one match:** {non_null_matches}"
                    )

                # Show average scores if available
                score_cols = [
                    col for col in results.columns if col.endswith('_score')
                ]
                if score_cols:
                    st.write("**Average match scores:**")
                    for score_col in score_cols:
                        avg_score = results[score_col].mean()
                        if pd.notna(avg_score):
                            match_num = score_col.replace('match', '').replace(
                                '_score', '')
                            st.write(f"  - Match {match_num}: {avg_score:.3f}")

    else:
        # Show message when no file is selected
        if data_source == "Upload your own file":
            st.info("👆 Please upload a file to continue")
        else:
            st.error("❌ Demo file is not available")

    # Information section
    with st.expander("About MetaHarmonizer"):
        st.markdown("""
        **MetaHarmonizer** is a tool for harmonizing clinical data schemas across different datasets.
        
        **Features:**
        - Automatic and manual schema mapping modes
        - Configurable number of top matches
        - Support for various file formats (CSV, TSV)
        - Toggle between detailed and simplified result views
        - Select specific columns for mapping
        
        **Parameters:**
        - **Mode**: 
          - `manual`: Interactive mapping process
          - `auto`: Automatic mapping based on similarity
        - **Top K**: Number of top matching suggestions to return
        
        **View Modes:**
        - **Detailed View**: Shows all columns including matched_stage, matched_stage_detail, and all match scores
        - **Simple View**: Shows only original_column and match fields for easy comparison
        
        **Repository:** [MetaHarmonizer on GitHub](https://github.com/shbrief/MetaHarmonizer)
        """)


if __name__ == "__main__":
    main()
