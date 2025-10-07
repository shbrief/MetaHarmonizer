import streamlit as st
import pandas as pd
import json
from typing import List, Dict
import plotly.express as px

# Import the ontology mapping engine (assuming it's available)
try:
    from src.Engine import get_ontology_engine
    OntoMapEngine = get_ontology_engine()
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False

# Page configuration
st.set_page_config(page_title="Ontology Mapper",
                   page_icon="🧬",
                   layout="wide",
                   initial_sidebar_state="expanded")

if not ENGINE_AVAILABLE:
    st.warning("⚠️ Ontology mapping engine not found. Running in demo mode.")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .result-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""",
            unsafe_allow_html=True)

# Initialize session state
if 'mapping_results' not in st.session_state:
    st.session_state.mapping_results = None
if 'query_terms' not in st.session_state:
    st.session_state.query_terms = []
if 'corpus_terms' not in st.session_state:
    st.session_state.corpus_terms = []


def load_sample_data():
    """Load sample data for demonstration"""
    sample_queries = {
        'disease': ["Breast Cancer", "Lung Carcinoma", "Melanoma", "Leukemia"],
        'body_site': ["Heart", "Liver", "Brain", "Kidney"],
        'treatment': ["Abiraterone", "Adalimumab", "17-DMAG", "Paclitaxel"]
    }

    sample_corpus = {
        'disease': [
            "Malignant neoplasm of breast", "Carcinoma of lung",
            "Malignant melanoma", "Acute lymphoblastic leukemia"
        ],
        'body_site': [
            "Cardiac muscle tissue", "Hepatic tissue", "Brain tissue",
            "Renal tissue"
        ],
        'treatment': [
            "Abiraterone acetate", "Adalimumab injection",
            "17-Dimethylaminoethylamino-17-demethoxygeldanamycin",
            "Paclitaxel injection"
        ]
    }

    return sample_queries, sample_corpus


def create_demo_results(query_terms: List[str], category: str) -> pd.DataFrame:
    """Create demo results when the actual engine is not available"""
    import random

    results = []
    for term in query_terms:
        # Simulate mapping results
        score1 = random.uniform(0.85, 0.99)
        score2 = random.uniform(0.70, 0.84)
        score3 = random.uniform(0.55, 0.69)

        result = {
            'original_value': term,
            'top1_match': f"Standardized {term}",
            'top1_score': score1,
            'top2_match': f"Alternative {term}",
            'top2_score': score2,
            'top3_match': f"Related {term}",
            'top3_score': score3,
            'match_level': 1 if score1 > 0.9 else 2 if score1 > 0.8 else 3,
            'stage': 'semantic_matching'
        }
        results.append(result)

    return pd.DataFrame(results)


def run_ontology_mapping(query_terms: List[str], corpus_terms: List[str],
                         category: str, method: str, om_strategy: str,
                         topk: int, test_or_prod: str,
                         cura_map: Dict) -> pd.DataFrame:
    """Run the ontology mapping engine or return demo results"""

    if not ENGINE_AVAILABLE:
        st.info("🔄 Running in demo mode with simulated results...")
        return create_demo_results(query_terms, category)

    try:
        # Initialize the engine
        onto_engine = OntoMapEngine(method=method,
                                    category=category,
                                    topk=topk,
                                    query=query_terms,
                                    corpus=corpus_terms,
                                    cura_map=cura_map,
                                    om_strategy=om_strategy,
                                    test_or_prod=test_or_prod)

        # Run the mapping
        with st.spinner("Running ontology mapping..."):
            results = onto_engine.run()

        return results

    except Exception as e:
        st.error(f"Error running ontology mapping: {str(e)}")
        st.info("Falling back to demo mode...")
        return create_demo_results(query_terms, category)


def display_results_summary(results_df: pd.DataFrame):
    """Display a summary of mapping results"""
    if results_df is None or results_df.empty:
        return

    st.markdown('<div class="section-header">📊 Mapping Summary</div>',
                unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    scores = pd.to_numeric(results_df.get('top1_score',
                                          pd.Series(dtype=float)),
                           errors='coerce').dropna()

    total_terms = len(results_df)
    high_confidence = int((scores >= 0.9).sum())
    medium_confidence = int(((scores >= 0.7) & (scores < 0.9)).sum())
    low_confidence = int((scores < 0.7).sum())

    with col1:
        st.metric("Total Terms", total_terms)
    with col2:
        st.metric("High Confidence (≥0.9)", high_confidence)
    with col3:
        st.metric("Medium Confidence (0.7–0.9)", medium_confidence)
    with col4:
        st.metric("Low Confidence (<0.7)", low_confidence)


def create_confidence_distribution_chart(results_df: pd.DataFrame):
    """Create a confidence score distribution chart"""
    if results_df is None or results_df.empty:
        return None

    fig = px.histogram(results_df,
                       x='top1_score',
                       nbins=20,
                       title="Distribution of Top Match Confidence Scores",
                       labels={
                           'top1_score': 'Confidence Score',
                           'count': 'Number of Terms'
                       },
                       color_discrete_sequence=['#1f77b4'])

    fig.update_layout(xaxis=dict(range=[0, 1]), showlegend=False)

    return fig


def create_match_level_chart(results_df: pd.DataFrame):
    """Create a match level distribution chart"""
    if results_df is None or results_df.empty:
        return None

    match_level_counts = results_df['match_level'].value_counts().sort_index()

    fig = px.bar(x=match_level_counts.index,
                 y=match_level_counts.values,
                 title="Distribution of Match Levels",
                 labels={
                     'x': 'Match Level',
                     'y': 'Number of Terms'
                 },
                 color=match_level_counts.values,
                 color_continuous_scale='Viridis')

    return fig


# Main app
def main():
    st.markdown('<div class="main-header">🧬 Ontology Mapper</div>',
                unsafe_allow_html=True)
    st.markdown(
        "**Standardize your data values using ontology mapping with NCIt corpus**"
    )

    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")

    # Category selection
    category = st.sidebar.selectbox("Select Category",
                                    ["disease", "body_site", "treatment"],
                                    help="Type of terms you're mapping")

    # Method selection
    method = st.sidebar.selectbox("Mapping Method", ["mt-sap-bert"],
                                  help="The mapping algorithm to use")

    # Strategy selection
    om_strategy = st.sidebar.selectbox(
        "Matching Strategy", ["st", "lm", "rag"],
        help=
        "st: Sentence-transformer (recommended), lm: CLS-token, rag: Context-enriched"
    )

    # Parameters
    topk = st.sidebar.slider("Top K Results",
                             min_value=1,
                             max_value=20,
                             value=5,
                             help="Number of candidates to return")
    test_or_prod = st.sidebar.selectbox(
        "Mode", ["prod", "test"],
        help="prod: Final mappings, test: Include evaluation metrics")

    # Load sample data
    sample_queries, sample_corpus = load_sample_data()

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(
        ["📝 Input Data", "🔄 Run Mapping", "📊 Results & Analysis"])

    with tab1:
        st.markdown('<div class="section-header">Input Your Terms</div>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Query Terms (Terms to Standardize)")

            # Option to use sample data
            if st.button(f"Load Sample {category.title()} Terms"):
                st.session_state.query_terms = sample_queries[category]

            # Text area for query terms
            query_text = st.text_area(
                "Enter terms (one per line):",
                value="\n".join(st.session_state.query_terms),
                height=200,
                help="Enter the terms you want to standardize, one per line")

            # File upload for query terms
            query_file = st.file_uploader("Or upload a file (CSV/TSV)",
                                          type=['csv', 'tsv'],
                                          key="query_file")

            if query_file:
                if query_file.name.endswith('.csv'):
                    df = pd.read_csv(query_file)
                    if not df.empty:
                        # Use the first column
                        query_terms = df.iloc[:,
                                              0].dropna().astype(str).tolist()
                        st.session_state.query_terms = query_terms
                        st.success(
                            f"Loaded {len(query_terms)} query terms from CSV")
                else:
                    content = query_file.read().decode('utf-8')
                    query_terms = [
                        line.strip() for line in content.split('\n')
                        if line.strip()
                    ]
                    st.session_state.query_terms = query_terms
                    st.success(
                        f"Loaded {len(query_terms)} query terms from TSV")

            # Update query terms from text area
            if query_text:
                st.session_state.query_terms = [
                    line.strip() for line in query_text.split('\n')
                    if line.strip()
                ]

            st.info(
                f"Current query terms: {len(st.session_state.query_terms)}")

        with col2:
            st.subheader("Corpus Terms (Standard Ontology)")

            # Option to use sample corpus
            if st.button(f"Load Sample {category.title()} Corpus"):
                st.session_state.corpus_terms = sample_corpus[category]

            # Text area for corpus terms
            corpus_text = st.text_area(
                "Enter standard terms (one per line):",
                value="\n".join(st.session_state.corpus_terms),
                height=200,
                help=
                "Enter the standard ontology terms to map against, one per line"
            )

            # File upload for corpus terms
            corpus_file = st.file_uploader("Or upload a file (CSV/TSV)",
                                           type=['csv', 'tsv'],
                                           key="corpus_file")

            if corpus_file:
                if corpus_file.name.endswith('.csv'):
                    df = pd.read_csv(corpus_file)
                    if not df.empty:
                        corpus_terms = df.iloc[:, 0].dropna().astype(
                            str).tolist()
                        st.session_state.corpus_terms = corpus_terms
                        st.success(
                            f"Loaded {len(corpus_terms)} corpus terms from CSV"
                        )
                else:
                    content = corpus_file.read().decode('utf-8')
                    corpus_terms = [
                        line.strip() for line in content.split('\n')
                        if line.strip()
                    ]
                    st.session_state.corpus_terms = corpus_terms
                    st.success(
                        f"Loaded {len(corpus_terms)} corpus terms from TSV")

            # Update corpus terms from text area
            if corpus_text:
                st.session_state.corpus_terms = [
                    line.strip() for line in corpus_text.split('\n')
                    if line.strip()
                ]

            st.info(
                f"Current corpus terms: {len(st.session_state.corpus_terms)}")

        # Curated mappings section
        st.markdown(
            '<div class="section-header">🎯 Curated Mappings (Optional)</div>',
            unsafe_allow_html=True)
        st.markdown("Provide known mappings to improve accuracy:")

        cura_map_text = st.text_area(
            "Enter curated mappings as JSON:",
            value='{}',
            help=
            'Example: {"old_term": "new_term", "another_term": "standard_term"}'
        )

    with tab2:
        st.markdown('<div class="section-header">Run Ontology Mapping</div>',
                    unsafe_allow_html=True)

        # Display current configuration
        st.subheader("Current Configuration")
        config_col1, config_col2 = st.columns(2)

        with config_col1:
            st.write(f"**Category:** {category}")
            st.write(f"**Method:** {method}")
            st.write(f"**Strategy:** {om_strategy}")

        with config_col2:
            st.write(f"**Top K:** {topk}")
            st.write(f"**Mode:** {test_or_prod}")
            st.write(f"**Query Terms:** {len(st.session_state.query_terms)}")

        # Run mapping button
        if st.button("🚀 Run Ontology Mapping",
                     type="primary",
                     disabled=len(st.session_state.query_terms) == 0):
            if len(st.session_state.query_terms) == 0:
                st.error("Please provide query terms to map")
            elif len(st.session_state.corpus_terms) == 0:
                st.error("Please provide corpus terms for mapping")
            else:
                try:
                    # Parse curated mappings
                    cura_map = json.loads(
                        cura_map_text) if cura_map_text.strip() else {}

                    # Run the mapping
                    results = run_ontology_mapping(
                        st.session_state.query_terms,
                        st.session_state.corpus_terms, category, method,
                        om_strategy, topk, test_or_prod, cura_map)

                    st.session_state.mapping_results = results
                    st.success("✅ Mapping completed successfully!")

                except json.JSONDecodeError:
                    st.error("Invalid JSON format in curated mappings")
                except Exception as e:
                    st.error(f"Error during mapping: {str(e)}")

    with tab3:
        if st.session_state.mapping_results is not None:
            results_df = st.session_state.mapping_results

            # Summary
            display_results_summary(results_df)

            # Charts
            st.markdown('<div class="section-header">📈 Visualizations</div>',
                        unsafe_allow_html=True)

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                confidence_chart = create_confidence_distribution_chart(
                    results_df)
                if confidence_chart:
                    st.plotly_chart(confidence_chart, use_container_width=True)

            with chart_col2:
                match_level_chart = create_match_level_chart(results_df)
                if match_level_chart:
                    st.plotly_chart(match_level_chart,
                                    use_container_width=True)

            # Detailed results
            st.markdown('<div class="section-header">📋 Detailed Results</div>',
                        unsafe_allow_html=True)

            # Filter options
            filter_col1, filter_col2 = st.columns(2)

            with filter_col1:
                min_confidence = st.slider("Minimum Confidence Score", 0.0,
                                           1.0, 0.0, 0.1)

            with filter_col2:
                show_columns = st.multiselect("Select Columns to Display",
                                              results_df.columns.tolist(),
                                              default=[
                                                  'original_value',
                                                  'top1_match', 'top1_score',
                                                  'match_level'
                                              ])

            # Filter results
            results_df['top1_score'] = pd.to_numeric(results_df['top1_score'],
                                                     errors='coerce')
            filtered_results = results_df[results_df['top1_score'] >=
                                          min_confidence]

            if show_columns:
                filtered_results = filtered_results[show_columns]

            # Display results
            st.dataframe(filtered_results, use_container_width=True)

            # Download options
            st.markdown('<div class="section-header">💾 Download Results</div>',
                        unsafe_allow_html=True)

            download_col1, download_col2 = st.columns(2)

            with download_col1:
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"ontology_mapping_results_{category}.csv",
                    mime="text/csv")

            with download_col2:
                json_data = results_df.to_json(orient='records', indent=2)
                st.download_button(
                    label="📥 Download as JSON",
                    data=json_data,
                    file_name=f"ontology_mapping_results_{category}.json",
                    mime="application/json")

            # Individual result cards for high-confidence matches
            st.markdown(
                '<div class="section-header">🎯 High Confidence Matches</div>',
                unsafe_allow_html=True)

            high_conf_results = results_df[results_df['top1_score'] >= 0.9]

            if not high_conf_results.empty:
                for _, row in high_conf_results.iterrows():
                    st.markdown(f"""
                    <div class="result-card">
                        <h4>{row['original_value']} → {row['top1_match']}</h4>
                        <p><strong>Confidence:</strong> {row['top1_score']:.3f}</p>
                        <p><strong>Match Level:</strong> {row['match_level']}</p>
                        <p><strong>Stage:</strong> {row['stage']}</p>
                    </div>
                    """,
                                unsafe_allow_html=True)
            else:
                st.info(
                    "No high-confidence matches found. Consider adjusting your corpus or query terms."
                )

        else:
            st.info(
                "👆 Run the ontology mapping in the 'Run Mapping' tab to see results here"
            )

    # Footer
    st.markdown("---")
    st.markdown(
        "**Ontology Mapper** - Standardize your data values using advanced semantic matching"
    )


if __name__ == "__main__":
    main()
