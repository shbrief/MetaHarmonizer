from src.models import ontology_mapper_st as oms
from src.models import ontology_mapper_lm as oml
from src.models import ontology_mapper_rag_faiss as omr
from src.models import ontology_mapper_bi_encoder as ombe
import pandas as pd
import numpy as np
from thefuzz import fuzz
from src.CustomLogger.custom_logger import CustomLogger

logger = CustomLogger()


class OntoMapEngine:
    """
    A class to initialize and run the OntoMapEngine for ontology mapping.

    Attributes:
        method (str): The name of the method.
        query (list[str]): The list of queries.
        corpus (list[str]): The list of corpus.
        cura_map (dict): The dictionary containing the mapping of queries to curated values.
        topk (int): The number of top matches to return.
        yaml_path (str): The path to the YAML file.
        om_strategy (str): The strategy to use for OntoMap.
        other_params (dict): Other parameters to pass to the engine.
        _test_or_prod (str): Indicates whether the environment is test or production.
        _logger (CustomLogger): Logger instance.
    """

    def __init__(self,
                 method: str,
                 category: str,
                 query: list[str],
                 corpus: list[str],
                 cura_map: dict,
                 topk: int = 5,
                 om_strategy: str = 'lm',
                 **other_params: dict) -> None:
        """
        Initializes the OntoMapEngine class.

        Args:
            method (str): The name of the method.
            query (list[str]): The list of queries.
            corpus (list[str]): The list of corpus.
            cura_map (dict): The dictionary containing the mapping of queries to curated values.
            topk (int, optional): The number of top matches to return. Defaults to 5.
            om_strategy (str, optional): The strategy to use for OntoMap. Defaults to 'lm'.
            **other_params (dict): Other parameters to pass to the engine.
        """
        self.method = method
        self.query = query
        self.category = category
        self.corpus = list(
            dict.fromkeys(corpus))  # Remove duplicates while preserving order
        self.topk = topk
        self.om_strategy = om_strategy
        self.cura_map = cura_map
        self.other_params = other_params
        if 'test_or_prod' not in self.other_params.keys():
            raise ValueError(
                "test_or_prod value must be defined in other_params dictionary"
            )

        self._test_or_prod = self.other_params['test_or_prod']
        self._logger = logger.custlogger(loglevel='INFO')
        self._logger.info("Initialized OntoMap Engine module")

    def _exact_matching(self):
        """
        Performs exact matching of queries to the corpus.

        Returns:
            list: The list of exact matches from the query.
        """
        # corpus_lower = {c.lower() for c in self.corpus}
        # return [q for q in self.query if q.lower() in corpus_lower]

        return [q for q in self.query if q in self.corpus]

    def _fuzzy_matching(self, fuzz_ratio: int = 80):
        """
        Performs fuzzy matching of queries to the corpus.

        Args:
            fuzz_ratio (int, optional): The fuzz ratio threshold for matching. Defaults to 80.

        Returns:
            list[str]: The list of fuzzy matches from the query.
        """
        return [
            q for q in self.query
            if fuzz.partial_ratio(q, self.corpus) > fuzz_ratio
        ]

    def _map_shortname_to_fullname(self, non_exact_list: list[str]) -> dict:
        """
        Return a dict: original_value -> updated_value
        (short name (code) → full name if exists, otherwise keep original)
        """
        mapping_df = pd.read_csv("data/corpus/oncotree_code_to_name.csv")
        short_to_name = dict(
            zip(mapping_df["code"].str.strip(),
                mapping_df["name"].str.strip()))

        replaced = {}
        for q in non_exact_list:
            q_strip = q.strip()
            replaced[q] = short_to_name.get(q_strip, q_strip)
            if q_strip in short_to_name:
                self._logger.info(
                    f"Replaced: {q_strip} → {short_to_name[q_strip]}")
        return replaced

    def _om_model_from_strategy(self, non_exact_query_list: list[str]):
        """
        Returns the OntoMap model based on the strategy.

        Args:
            non_exact_query_list (list[str]): The list of non-exact query strings.

        Returns:
            object: The OntoMap model instance.
        """
        query_df = self.other_params.get('query_df', None)
        corpus_df = self.other_params.get('corpus_df', None)

        if self.om_strategy == 'lm':
            return oml.OntoMapLM(method=self.method,
                                 category=self.category,
                                 om_strategy=self.om_strategy,
                                 query=non_exact_query_list,
                                 corpus=self.corpus,
                                 topk=self.topk,
                                 from_tokenizer=True)

        elif self.om_strategy == 'st':
            return oms.OntoMapST(method=self.method,
                                 category=self.category,
                                 om_strategy=self.om_strategy,
                                 query=non_exact_query_list,
                                 corpus=self.corpus,
                                 topk=self.topk,
                                 from_tokenizer=False)
        elif self.om_strategy == 'rag':
            return omr.OntoMapRAG(method=self.method,
                                  category=self.category,
                                  om_strategy=self.om_strategy,
                                  query=non_exact_query_list,
                                  corpus=self.corpus,
                                  topk=self.topk,
                                  corpus_df=corpus_df)
        elif self.om_strategy == 'bie':
            return ombe.OntoMapBIE(method=self.method,
                                   category=self.category,
                                   om_strategy=self.om_strategy,
                                   query=non_exact_query_list,
                                   corpus=self.corpus,
                                   topk=self.topk,
                                   query_df=query_df,
                                   corpus_df=corpus_df)
        else:
            raise ValueError(
                "om_strategy should be either 'st', 'lm', 'rag', or 'bie'")

    def _separate_matches(self, matching_type: str = 'exact'):
        """
        Separates exact and non-exact matches.

        Args:
            matching_type (str, optional): The type of matching to perform ('exact' or 'fuzzy'). Defaults to 'exact'.

        Returns:
            list: The list of non-exact matches.
        """
        if matching_type not in ['exact', 'fuzzy']:
            raise ValueError(
                "Matching type should be either 'exact' or 'fuzzy'")

        if matching_type == 'exact':
            to_separate_matches = self._exact_matching()
        elif matching_type == 'fuzzy':
            to_separate_matches = self._fuzzy_matching()

        non_exact_matches = list(np.setdiff1d(self.query, to_separate_matches))
        return non_exact_matches

    def get_results_for_non_exact(self,
                                  non_exact_query_list: list[str],
                                  topk: int = 5):
        """
        Retrieves the match results for the given non-exact query list.

        Args:
            non_exact_query_list (list[str]): The list of non-exact query strings.
            topk (int, optional): The number of top matches to retrieve. Defaults to 5.

        Returns:
            pd.DataFrame: The DataFrame containing the match results.
        """
        onto_map = self._om_model_from_strategy(
            non_exact_query_list=non_exact_query_list)
        return onto_map.get_match_results(cura_map=self.cura_map,
                                          topk=topk,
                                          test_or_prod=self._test_or_prod)

    def run(self):
        """
        Runs the OntoMap Engine module.

        Returns:
            pd.DataFrame: A DataFrame containing both exact and non-exact matches.
        """
        self._logger.info("Running Ontology Mapping")
        self._logger.info("Separating exact and non-exact matches")
        exact_matches = self._exact_matching()
        non_exact_matches_ls = self._separate_matches(matching_type='exact')

        # self._logger.info("Running OntoMap model for non-exact matches")
        # onto_map_res = self.get_results_for_non_exact(
        #     non_exact_query_list=non_exact_matches_ls, topk=self.topk)

        self._logger.info("Replacing shortNames using rule-based name mapping")
        mapping_dict = self._map_shortname_to_fullname(non_exact_matches_ls)
        updated_queries = [mapping_dict[q] for q in non_exact_matches_ls]

        if updated_queries:
            replace_df = pd.DataFrame({
                "original_value": non_exact_matches_ls,
                "updated_value": updated_queries
            })

            updated_cura_map = {
                mapping_dict[k]: v
                for k, v in self.cura_map.items() if k in mapping_dict
            }

            onto_map = self._om_model_from_strategy(updated_queries)
            onto_map_res = onto_map.get_match_results(
                cura_map=updated_cura_map,
                topk=self.topk,
                test_or_prod=self._test_or_prod)

            onto_map_res.rename(columns={"original_value": "updated_value"},
                                inplace=True)
            onto_map_res = pd.merge(replace_df,
                                    onto_map_res,
                                    on="updated_value",
                                    how="left")
            onto_map_res["curated_ontology"] = onto_map_res[
                "original_value"].map(self.cura_map).fillna("Not Found")
        else:
            onto_map_res = self.get_results_for_non_exact(
                non_exact_query_list=non_exact_matches_ls, topk=self.topk)

        # Create DataFrame for exact matches
        exact_df = pd.DataFrame({'original_value': exact_matches})
        exact_df['curated_ontology'] = exact_df[
            'original_value']  # For exact matches, these are the same
        exact_df['match_level'] = 1
        exact_df['stage'] = 1
        for i in range(1, self.topk + 1):
            exact_df[f'top{i}_match'] = exact_df['curated_ontology']
            exact_df[f'top{i}_score'] = 1.00

        # Add stage column to onto_map_res
        onto_map_res['stage'] = 2

        # Combine exact matches and non-exact matches
        combined_results = pd.concat([exact_df, onto_map_res],
                                     ignore_index=True)

        return combined_results
