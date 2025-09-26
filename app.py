import streamlit as st
import pandas as pd
import tempfile
import os
from io import StringIO
import sys

# Add error handling for the import
try:
    from src.Engine import SchemaMapEngine
    METAHARMONIZER_AVAILABLE = True
    import_error = None
except ImportError as e:
    METAHARMONIZER_AVAILABLE = False
    import_error = str(e)

def show_setup_instructions():
    """Display comprehensive setup instructions"""
    st.error(f"""
    ❌ **MetaHarmonizer not found!**
    
    Error: {import_error}
    """)
    
    st.markdown("""
    ## 🔧 Setup Instructions
    
    ### Option 1: Quick Fix (if you have the repository)
    If you already have MetaHarmonizer cloned, the issue might be a missing dependency:
    
    ```bash
    pip install faiss-cpu
    # OR for GPU support:
    pip install faiss-gpu
    ```
    
    ### Option 2: Full Setup
    """)
    
    with st.expander("📋 Complete Setup Steps", expanded=True):
        st.markdown("""
        **Step 1: Clone the repository**
        ```bash
        git clone https://github.com/shbrief/MetaHarmonizer
        cd MetaHarmonizer
        ```
        
        **Step 2: Install dependencies**
        ```bash
        pip install -r requirements.txt
        ```
        
        **Step 3: Install FAISS (if not in requirements.txt)**
        ```bash
        pip install faiss-cpu
        ```
        
        **Step 4: Run the app from MetaHarmonizer directory**
        ```bash
        streamlit run path/to/your/app.py
        ```
        """)
    
    with st.expander("🐛 Troubleshooting Common Issues"):
        st.markdown("""
        **Issue: "No module named 'faiss'"**
        - Solution: `pip install faiss-cpu`
        - For GPU: `pip install faiss-gpu`
        
        **Issue: "No module named 'src'"**
        - Make sure you're running the app from within the MetaHarmonizer directory
        - Or add the MetaHarmonizer directory to your Python path
        
        **Issue: FAISS installation fails**
        - Try: `pip install faiss-cpu --no-cache-dir`
        - Make sure you're using Python 3.7-3.11
        - Consider using conda: `conda install -c conda-forge faiss-cpu`
        
        **Issue: Other missing dependencies**
        - Run: `pip install -r requirements.txt` from the MetaHarmonizer directory
        
        **For Streamlit Cloud deployment:**
        - Ensure your `requirements.txt` includes all dependencies:
          ```
          faiss-cpu
          pandas
          streamlit
          # ... other dependencies
          ```
        """)
    
    with st.expander("📁 Directory Structure Check"):
        st.markdown("""
        Your directory structure should look like:
        ```
        MetaHarmonizer/
        ├── src/
        │   ├── Engine.py
        │   └── ...
        ├── data/
        │   └── demo_data/
        │       └── clinical_metadata_demo.tsv
        ├── requirements.txt
        ├── your_app.py  # This Streamlit app
        └── ...
        ```
        """)

def main():
    st.set_page_config(
        page_title="MetaHarmonizer Schema Mapping",
        page_icon="🔗",
        layout="wide"
    )
    
    st.title("🔗 MetaHarmonizer Schema Mapping Tool")
    st.markdown("Upload your clinical data file and run schema mapping using MetaHarmonizer")
    
    # Check if MetaHarmonizer is available
    if not METAHARMONIZER_AVAILABLE:
        show_setup_instructions()
        
        # Add a refresh button
        if st.button("🔄 Refresh App", help="Click after installing dependencies"):
            st.rerun()
        
        return
    
    # Show success message when everything is working
    st.success("✅ MetaHarmonizer is ready!")
    
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
    
    # Radio button to choose between demo or upload
    data_source = st.radio(
        "Select data source:",
        options=["Upload your own file", "Use demo file"],
        help="Choose whether to upload your own data or test with the demo file"
    )
    
    demo_file_path = "data/demo_data/clinical_metadata_demo.tsv"
    
    if data_source == "Use demo file":
        # Demo file section
        st.subheader("🎯 Demo File Selected")
        
        if os.path.exists(demo_file_path):
            try:
                demo_df = pd.read_csv(demo_file_path, sep='\t')
                st.success(f"✅ Demo file loaded: `{demo_file_path}`")
                st.write(f"**Shape:** {demo_df.shape[0]} rows × {demo_df.shape[1]} columns")
                
                with st.expander("👀 Preview demo data"):
                    st.dataframe(demo_df.head(10))
                    if len(demo_df) > 10:
                        st.info(f"Showing first 10 rows of {len(demo_df)} total rows")
                
                # Set variables for processing
                selected_file = demo_file_path
                file_available = True
                
            except Exception as e:
                st.error(f"Could not load demo file: {e}")
                file_available = False
                selected_file = None
        else:
            st.error(f"Demo file not found at: `{demo_file_path}`")
            st.info("💡 Make sure you're running this app from the MetaHarmonizer directory")
            file_available = False
            selected_file = None
    
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
            
            # Show file preview
            with st.expander("👀 Preview uploaded data"):
                try:
                    # Determine separator based on file extension
                    if uploaded_file.name.endswith('.tsv'):
                        separator = '\t'
                    else:
                        separator = ','
                    
                    # Read and display preview
                    df_preview = pd.read_csv(uploaded_file, sep=separator, nrows=5)
                    st.dataframe(df_preview)
                    st.info(f"File shape: {df_preview.shape[0]}+ rows × {df_preview.shape[1]} columns")
                    
                    # Reset file pointer for processing
                    uploaded_file.seek(0)
                except Exception as e:
                    st.error(f"Error reading file: {e}")
                    file_available = False
                    selected_file = None
                    return
            
            # Set variables for processing
            selected_file = uploaded_file
            file_available = True
        else:
            file_available = False
            selected_file = None
    
    # Processing section (only show if file is available)
    if file_available:
        # Processing section
        st.header("🔄 Schema Mapping")
        
        if st.button("🚀 Run Schema Mapping", type="primary"):
            with st.spinner("Processing schema mapping... This may take a few minutes."):
                try:
                    # Handle file processing based on source
                    if data_source == "Use demo file":
                        # Use demo file directly
                        tmp_file_path = selected_file
                        cleanup_needed = False
                        st.info("Processing demo file...")
                    else:
                        # Create temporary file for uploaded file
                        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.tsv') as tmp_file:
                            # Write uploaded content to temporary file
                            content = selected_file.read()
                            if isinstance(content, bytes):
                                content = content.decode('utf-8')
                            tmp_file.write(content)
                            tmp_file_path = tmp_file.name
                        cleanup_needed = True
                        st.info("Processing uploaded file...")
                    
                    # Initialize the SchemaMapEngine
                    engine = SchemaMapEngine(
                        clinical_data_path=tmp_file_path,
                        mode=mode,
                        top_k=top_k
                    )
                    
                    # Run schema mapping
                    results = engine.run_schema_mapping()

                    # Remove `_source` columns
                    pattern = '_source'
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
                    
                    file_suffix = "demo" if data_source == "Use demo file" else "uploaded"
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_buffer.getvalue(),
                        file_name=f"schema_mapping_results_{file_suffix}_{mode}_top{top_k}.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error during schema mapping: {str(e)}")
                    
                    # More detailed error info in expander
                    with st.expander("🔍 Detailed Error Information"):
                        st.code(str(e))
                        st.write("**Possible solutions:**")
                        st.write("- Check if your file format is correct")
                        st.write("- Ensure the file contains valid clinical data")
                        st.write("- Try a different mode (manual/auto)")
                        st.write("- Check if all MetaHarmonizer dependencies are installed")
                        st.write("- Make sure FAISS is installed: `pip install faiss-cpu`")
    
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
        - Support for various file formats (CSV, TSV, TXT)
        
        **Parameters:**
        - **Mode**: 
          - `manual`: Interactive mapping process
          - `auto`: Automatic mapping based on similarity
        - **Top K**: Number of top matching suggestions to return
        
        **Repository:** [MetaHarmonizer on GitHub](https://github.com/shbrief/MetaHarmonizer)
        """)

if __name__ == "__main__":
    main()
