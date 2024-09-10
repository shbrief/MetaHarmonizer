import pandas as pd
import numpy as np
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
import src.models.ontology_models as otm
import src.CustomLogger.custom_logger
from tqdm import tqdm
class SapBertEmbeddings(Embeddings):
    def __init__(self, sapbert_model):
        self.model = sapbert_model

    def embed_documents(self, texts):
        return [self.model.create_embeddings(text)[0].tolist() for text in texts]

    def embed_query(self, text):
        return self.model.create_embeddings(text)[0].tolist()

class OntoMapRAG(otm.OntoModelsBase):
    def __init__(self, method: str, query: list[str], corpus: list[str], topk: int = 5, 
                 collection=None, embedding_model=None, index_name="vector_search_index", 
                 embedding_key="embeddings", text_key="context", yaml_path: str = 'method_model.yaml'):
        super().__init__(method, topk, query, corpus, yaml_path)
        self.collection = collection
        self.embedding_model = embedding_model
        self.index_name = index_name
        self.embedding_key = embedding_key
        self.text_key = text_key
        self.vector_store = None
        self.logger = src.CustomLogger.custom_logger.CustomLogger().custlogger(loglevel='INFO')
        self.logger.info("Initialized OntoMapRAG module")

    def initialize_vector_store(self):
        if self.method == 'openai':
            embedding_model = OpenAIEmbeddings(disallowed_special=())
        elif self.method == 'sap-bert':
            embedding_model = SapBertEmbeddings(self.embedding_model)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

        self.vector_store = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=embedding_model,
            index_name=self.index_name,
            embedding_key=self.embedding_key,
            text_key=self.text_key
        )

    def get_match_results(self, cura_map: dict[str, str] = None, topk: int = 5, 
                        test_or_prod: str = 'test', oversampling_factor: int = 50) -> pd.DataFrame:
        if test_or_prod == 'test' and cura_map is None:
            raise ValueError("cura_map should be provided for test mode")

        if self.vector_store is None:
            self.initialize_vector_store()

        self.logger.info("Generating results table")
        
        results = []
        for query in tqdm(self.query, desc="Processing queries"):
            search_results = self.vector_store.similarity_search(
                query,
                k=topk,
                include_scores=True,
                oversampling_factor=oversampling_factor
            )
            results.append(search_results)
        
        # Create DataFrame from results
        df = pd.DataFrame({
            'original_value': self.query,
            'curated_ontology': [cura_map.get(q, "Not Found") if test_or_prod == 'test' else "Not Available for Prod Environment" for q in self.query]
        })
        
        # Add top matches and scores
        for i in range(topk):
            df[f'top{i+1}_match'] = [r[i].metadata['term'] if i < len(r) else "N/A" for r in results]
            df[f'top{i+1}_score'] = ["{:.4f}".format(r[i].metadata['score']) if i < len(r) else "N/A" for r in results]
        
        # Calculate match level
        df['match_level'] = df.apply(lambda row: next((i+1 for i, col in enumerate(f'top{j+1}_match' for j in range(topk)) 
                                                    if row[col] == row['curated_ontology']), 99), axis=1)
        
        self.logger.info("Results Generated")
        return df

    def evaluate_similarity_search(self, small_corpus_map):
        total_score = 0
        total_queries = len(self.query)

        for query in self.query:
            results = self.vector_store.similarity_search(query, k=self.topk)
            result_terms = [doc.metadata['term'] for doc in results]
            
            curated_term = small_corpus_map[small_corpus_map['original_value'] == query]['curated_ontology'].values
            
            if len(curated_term) > 0:
                curated_term = curated_term[0]
                match_score = 1 if curated_term in result_terms else 0
                total_score += match_score
            else:
                total_queries -= 1

        average_match_score = total_score / total_queries if total_queries > 0 else 0
        return average_match_score

    def max_marginal_relevance_search(self, query, k=10, fetch_k=50, lambda_mult=0.5):
        results = self.vector_store.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            include_scores=True,
        )
        return results