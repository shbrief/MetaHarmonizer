import streamlit as st
import pandas as pd
import tempfile
import os
from io import StringIO
<<<<<<< HEAD

# Add error handling for the import
try:
    from src.Engine import get_schema_engine
    SchemaMapEngine = get_schema_engine()
=======
import sys

# Add error handling for the import
try:
    from src.Engine import SchemaMapEngine
>>>>>>> c77c097 (updated schema (sm) and ontology (om) mapper streamlit apps)
    METAHARMONIZER_AVAILABLE = True
except ImportError as e:
    METAHARMONIZER_AVAILABLE = False
    import_error = str(e)

<<<<<<< HEAD

def main():
    st.set_page_config(page_title="MetaHarmonizer Schema Mapping",
                       page_icon="🔗",
                       layout="wide")

    st.title("🔗 MetaHarmonizer Schema Mapping Tool")
    st.markdown(
        "Upload your clinical data file and run schema mapping using MetaHarmonizer"
    )

=======
def main():
    st.set_page_config(
        page_title="MetaHarmonizer Schema Mapping",
        page_icon="🔗",
        layout="wide"
    )
    
    st.title("🔗 MetaHarmonizer Schema Mapping Tool")
    st.markdown("Upload your clinical data file and run schema mapping using MetaHarmonizer")
    
>>>>>>> c77c097 (updated schema (sm) and ontology (om) mapper streamlit apps)
    # Check if MetaHarmonizer is available
    if not METAHARMONIZER_AVAILABLE:
        st.error(f"""
        ❌ **MetaHarmonizer not found!**
        
        Error: {import_error}
        
        **Setup Instructions:**
        1. Clone the repository: `git clone https://github.com/shbrief/MetaHarmonizer`
<<<<<<< HEAD
        2. Install dependencies: `pip install -r requirements_sm.txt`
=======
        2. Install dependencies: `pip install -r requirements.txt`
>>>>>>> c77c097 (updated schema (sm) and ontology (om) mapper streamlit apps)
        3. Make sure the `src` folder is in your Python path
        
        Or run this app from within the MetaHarmonizer directory.
        """)
        return
<<<<<<< HEAD

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

=======
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    mode = st.sidebar.selectbox(
        "Mapping Mode",
        options=["manual", "auto"],
        help="Choose between manual or automatic schema mapping"
    )
    
    top_k = st.sidebar.slider(
        "Top K Matches",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of top matches to return"
    )
    
    # File selection section
    st.header("📁 Choose Your Data Source")
    
>>>>>>> c77c097 (updated schema (sm) and ontology (om) mapper streamlit apps)
    # Radio button to choose between demo or upload
    data_source = st.radio(
        "Select data source:",
        options=["Upload your own file", "Use demo file"],
        help="Choose whether to upload your own data or test with the demo file"
    )
<<<<<<< HEAD

    demo_file_path = "data/demo_data/clinical_metadata_demo.tsv"

    if data_source == "Use demo file":
        # Demo file section
        st.subheader("🎯 Demo File Selected")

=======
    
    demo_file_path = "data/demo_data/clinical_metadata_demo.tsv"
    
    if data_source == "Use demo file":
        # Demo file section
        st.subheader("🎯 Demo File Selected")
        
>>>>>>> c77c097 (updated schema (sm) and ontology (om) mapper streamlit apps)
        if os.path.exists(demo_file_path):
            try:
                demo_df = pd.read_csv(demo_file_path, sep='\t')
                st.success(f"✅ Demo file loaded: `{demo_file_path}`")
<<<<<<< HEAD
                st.write(
                    f"**Shape:** {demo_df.shape[0]} rows × {demo_df.shape[1]} columns"
                )

                with st.expander("👀 Preview demo data"):
                    st.dataframe(demo_df.head(10))
                    if len(demo_df) > 10:
                        st.info(
                            f"Showing first 10 rows of {len(demo_df)} total rows"
                        )

                # Set variables for processing
                selected_file = demo_file_path
                file_available = True

=======
                st.write(f"**Shape:** {demo_df.shape[0]} rows × {demo_df.shape[1]} columns")
                
                with st.expander("👀 Preview demo data"):
                    st.dataframe(demo_df.head(10))
                    if len(demo_df) > 10:
                        st.info(f"Showing first 10 rows of {len(demo_df)} total rows")
                
                # Set variables for processing
                selected_file = demo_file_path
                file_available = True
                
>>>>>>> c77c097 (updated schema (sm) and ontology (om) mapper streamlit apps)
            except Exception as e:
                st.error(f"Could not load demo file: {e}")
                file_available = False
                selected_file = None
        else:
            st.error(f"Demo file not found at: `{demo_file_path}`")
            file_available = False
            selected_file = None
<<<<<<< HEAD

    else:
        # File upload section
        st.subheader("📤 Upload Your File")

        uploaded_file = st.file_uploader(
            "Choose a clinical data file",
            type=['csv', 'tsv'],
            help="Upload a CSV or TSV file containing clinical data")

        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")

=======
    
    else:
        # File upload section
        st.subheader("📤 Upload Your File")
        
        uploaded_file = st.file_uploader(
            "Choose a clinical data file",
            type=['csv', 'tsv', 'txt'],
            help="Upload a CSV, TSV, or TXT file containing clinical data"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
>>>>>>> c77c097 (updated schema (sm) and ontology (om) mapper streamlit apps)
            # Show file preview
            with st.expander("👀 Preview uploaded data"):
                try:
                    # Determine separator based on file extension
                    if uploaded_file.name.endswith('.tsv'):
                        separator = '\t'
                    else:
                        separator = ','
<<<<<<< HEAD

                    # Read and display preview
                    df_preview = pd.read_csv(uploaded_file,
                                             sep=separator,
                                             nrows=5)
                    st.dataframe(df_preview)
                    st.info(
                        f"File shape: {df_preview.shape[0]}+ rows × {df_preview.shape[1]} columns"
                    )

=======
                    
                    # Read and display preview
                    df_preview = pd.read_csv(uploaded_file, sep=separator, nrows=5)
                    st.dataframe(df_preview)
                    st.info(f"File shape: {df_preview.shape[0]}+ rows × {df_preview.shape[1]} columns")
                    
>>>>>>> c77c097 (updated schema (sm) and ontology (om) mapper streamlit apps)
                    # Reset file pointer for processing
                    uploaded_file.seek(0)
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                    file_available = False
                    selected_file = None
                    return
<<<<<<< HEAD

=======
            
>>>>>>> c77c097 (updated schema (sm) and ontology (om) mapper streamlit apps)
            # Set variables for processing
            selected_file = uploaded_file
            file_available = True
        else:
            file_available = False
            selected_file = None
<<<<<<< HEAD

=======
    
>>>>>>> c77c097 (updated schema (sm) and ontology (om) mapper streamlit apps)
    # Processing section (only show if file is available)
    if file_available:
        # Processing section
        st.header("🔄 Schema Mapping")
<<<<<<< HEAD

        if st.button("🚀 Run Schema Mapping", type="primary"):
            with st.spinner(
                    "Processing schema mapping... This may take a few minutes."
            ):
                info_ph = st.empty()  # Placeholder for info messages
=======
        
        if st.button("🚀 Run Schema Mapping", type="primary"):
            with st.spinner("Processing schema mapping... This may take a few minutes."):
>>>>>>> c77c097 (updated schema (sm) and ontology (om) mapper streamlit apps)
                try:
                    # Handle file processing based on source
                    if data_source == "Use demo file":
                        # Use demo file directly
                        tmp_file_path = selected_file
                        cleanup_needed = False
<<<<<<< HEAD
                        info_ph.info("Processing demo file...")
                    else:
                        # Create temporary file for uploaded file
                        with tempfile.NamedTemporaryFile(
                                mode='w+', delete=False,
                                suffix='.tsv') as tmp_file:
=======
                        st.info("Processing demo file...")
                    else:
                        # Create temporary file for uploaded file
                        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.tsv') as tmp_file:
>>>>>>> c77c097 (updated schema (sm) and ontology (om) mapper streamlit apps)
                            # Write uploaded content to temporary file
                            content = selected_file.read()
                            if isinstance(content, bytes):
                                content = content.decode('utf-8')
                            tmp_file.write(content)
                            tmp_file_path = tmp_file.name
                        cleanup_needed = True
<<<<<<< HEAD
                        info_ph.info("Processing uploaded file...")

                    # Initialize the SchemaMapEngine
                    engine = SchemaMapEngine(clinical_data_path=tmp_file_path,
                                             mode=mode,
                                             top_k=top_k)

=======
                        st.info("Processing uploaded file...")
                    
                    # Initialize the SchemaMapEngine
                    engine = SchemaMapEngine(
                        clinical_data_path=tmp_file_path,
                        mode=mode,
                        top_k=top_k
                    )
                    
>>>>>>> c77c097 (updated schema (sm) and ontology (om) mapper streamlit apps)
                    # Run schema mapping
                    results = engine.run_schema_mapping()

                    # Remove `_source` columns
                    pattern = '_source'
<<<<<<< HEAD
                    columns_to_drop = results.columns[
                        results.columns.str.contains(pattern, case=False)]
                    results.drop(columns=columns_to_drop,
                                 inplace=True)  # drop the column

                    # Clean up temporary file if needed
                    if cleanup_needed:
                        os.unlink(tmp_file_path)

                    # Display results
                    st.success("✅ Schema mapping completed successfully!")

                    # Results section
                    st.header("📊 Results")

                    # Display results dataframe
                    st.subheader("Mapping Results")
                    st.dataframe(results, use_container_width=True)

                    # Download button
                    csv_buffer = StringIO()
                    results.to_csv(csv_buffer, index=False)

=======
                    columns_to_drop = results.columns[results.columns.str.contains(pattern, case=False)]
                    results.drop(columns=columns_to_drop, inplace=True) # drop the column
                    
                    # Clean up temporary file if needed
                    if cleanup_needed:
                        os.unlink(tmp_file_path)
                    
                    # Display results
                    st.success("✅ Schema mapping completed successfully!")
                    
                    # Results section
                    st.header("📊 Results")
                    
                    # Display results dataframe
                    st.subheader("Mapping Results")
                    st.dataframe(results, use_container_width=True)
                    
                    # Download button
                    csv_buffer = StringIO()
                    results.to_csv(csv_buffer, index=False)
                    
>>>>>>> c77c097 (updated schema (sm) and ontology (om) mapper streamlit apps)
                    file_suffix = "demo" if data_source == "Use demo file" else "uploaded"
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_buffer.getvalue(),
<<<<<<< HEAD
                        file_name=
                        f"schema_mapping_results_{file_suffix}_{mode}_top{top_k}.csv",
                        mime="text/csv")

=======
                        file_name=f"schema_mapping_results_{file_suffix}_{mode}_top{top_k}.csv",
                        mime="text/csv"
                    )
                    
>>>>>>> c77c097 (updated schema (sm) and ontology (om) mapper streamlit apps)
                    # # Display summary statistics
                    # with st.expander("📈 Summary Statistics"):
                    #     st.write(f"**Total mappings found:** {len(results)}")
                    #     if 'confidence_score' in results.columns:
                    #         st.write(f"**Average confidence score:** {results['confidence_score'].mean():.3f}")
                    #     if 'original_column' in results.columns:
                    #         st.write(f"**Unique columns mapped:** {results['original_column'].nunique()}")
<<<<<<< HEAD
                    #
=======
                    #     
>>>>>>> c77c097 (updated schema (sm) and ontology (om) mapper streamlit apps)
                    #     # Display column info
                    #     st.write("**Result columns:**")
                    #     for col in results.columns:
                    #         st.write(f"- {col}")
<<<<<<< HEAD

                except Exception as e:
                    st.error(f"❌ Error during schema mapping: {str(e)}")

=======
                    
                except Exception as e:
                    st.error(f"❌ Error during schema mapping: {str(e)}")
                    
>>>>>>> c77c097 (updated schema (sm) and ontology (om) mapper streamlit apps)
                    # More detailed error info in expander
                    with st.expander("🔍 Detailed Error Information"):
                        st.code(str(e))
                        st.write("**Possible solutions:**")
                        st.write("- Check if your file format is correct")
<<<<<<< HEAD
                        st.write(
                            "- Ensure the file contains valid clinical data")
                        st.write("- Try a different mode (manual/auto)")
                        st.write(
                            "- Check if all MetaHarmonizer dependencies are installed"
                        )

                finally:
                    info_ph.empty()

=======
                        st.write("- Ensure the file contains valid clinical data")
                        st.write("- Try a different mode (manual/auto)")
                        st.write("- Check if all MetaHarmonizer dependencies are installed")
    
>>>>>>> c77c097 (updated schema (sm) and ontology (om) mapper streamlit apps)
    else:
        # Show message when no file is selected
        if data_source == "Upload your own file":
            st.info("👆 Please upload a file to continue")
        else:
            st.error("❌ Demo file is not available")

    # Information section
    with st.expander("ℹ️ About MetaHarmonizer"):
        st.markdown("""
        **MetaHarmonizer** is a tool for harmonizing clinical data schemas across different datasets.
        
        **Features:**
        - Automatic and manual schema mapping modes
        - Configurable number of top matches
<<<<<<< HEAD
        - Support for various file formats (CSV, TSV)
=======
        - Support for various file formats (CSV, TSV, TXT)
>>>>>>> c77c097 (updated schema (sm) and ontology (om) mapper streamlit apps)
        
        **Parameters:**
        - **Mode**: 
          - `manual`: Interactive mapping process
          - `auto`: Automatic mapping based on similarity
        - **Top K**: Number of top matching suggestions to return
        
        **Repository:** [MetaHarmonizer on GitHub](https://github.com/shbrief/MetaHarmonizer)
        """)

<<<<<<< HEAD

=======
>>>>>>> c77c097 (updated schema (sm) and ontology (om) mapper streamlit apps)
if __name__ == "__main__":
    main()
