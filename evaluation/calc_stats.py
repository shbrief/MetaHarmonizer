import pandas as pd
import numpy as np


class CalcStats:

    def __init__(self) -> None:
        pass

    def check_required_columns(self, data, required_columns=None):
        """
        Check if the required columns exist in the DataFrame.

        Parameters:
        - data (pandas.DataFrame): The input data to be checked.

        Returns:
        - missing_columns (list): A list of columns that are missing from the DataFrame.
        """
        if not set(required_columns).issubset(data.columns):
            missing_columns = list(set(required_columns) - set(data.columns))
            raise ValueError(f"Missing required columns: {missing_columns}")
        return True

    def calc_accuracy(self, data, mode="top5", custom_top_k=None):
        """
        Calculate the accuracy of the model predictions for multiple top-k values.

        Parameters:
        - data (pandas.DataFrame): The input data containing the model predictions and the ground truth labels.
        - mode (str): Evaluation mode. Options:
            - "top5": Standard top-1, 3, 5 evaluation (default)
            - "rerank": Evaluate reranking candidate quality at top-1, 5, 20, 30, 50
        - custom_top_k (list, optional): Custom list of top-k values. Overrides mode if provided.

        Returns:
        - accuracy_df (pandas.DataFrame): A DataFrame containing the accuracy levels and their corresponding accuracies.
        """
        # Determine top-k list based on mode
        if custom_top_k is not None:
            top_k_list = custom_top_k
        elif mode == "top5":
            top_k_list = [1, 3, 5]
        elif mode == "rerank":
            top_k_list = [1, 5, 20, 30, 50]
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Choose 'top5' or 'rerank', or provide custom_top_k."
            )

        # Determine maximum k needed
        max_k = max(top_k_list)

        # Build required columns list - now using ref_match instead of curated_ontology
        required_columns = ["ref_match"]
        required_columns.extend([f"match{i}" for i in range(1, max_k + 1)])

        # Check if all required columns exist
        self.check_required_columns(data, required_columns)

        # Make a copy to avoid modifying original data
        data = data.copy()

        # Calculate accuracy for each top-k
        accuracy_results = []

        for k in top_k_list:
            # Get the column names for top-k matches
            top_k_columns = [f"match{i}" for i in range(1, k + 1)]

            # Calculate if ref_match is in top-k matches (dynamically computed)
            matches = data.apply(
                lambda row: row["ref_match"] in row[top_k_columns].values,
                axis=1
            )

            # Calculate percentage accuracy
            accuracy_pct = matches.mean() * 100

            accuracy_results.append({
                "Accuracy Level": f"Top {k} Match{'es' if k > 1 else ''}",
                "Accuracy": accuracy_pct,
                "k": k
            })

        # Create DataFrame
        accuracy_df = pd.DataFrame(accuracy_results)

        return accuracy_df

    def calc_confusion_matrix(self, data, match_type):
        """
        Calculate the confusion matrix for the model predictions.

        Parameters:
        - data (pandas.DataFrame): The input data containing the model predictions and the ground truth labels.
        - match_type (str): The type of match to use for the confusion matrix. Can be 'match1', 'match3', or 'match5'.

        Returns:
        - confusion_matrix_df (pandas.DataFrame): A DataFrame containing the confusion matrix values.
        """
        required_columns = ["ref_match", match_type]

        self.check_required_columns(data, required_columns)
        # Create a confusion matrix
        confusion_matrix = pd.crosstab(data["ref_match"],
                                    data[match_type])
        confusion_matrix_df = pd.DataFrame(confusion_matrix)

        return confusion_matrix_df

    def calc_f1_score(self, data, match_type):
        """
        Calculate the F1 score for the model predictions.

        Parameters:
        - data (pandas.DataFrame): The input data containing the model predictions and the ground truth labels.
        - match_type (str): The type of match to use for the F1 score calculation. Can be 'match1', 'match3', or 'match5'.

        Returns:
        - f1_score (float): The F1 score for the model predictions.
        """
        required_columns = ["ref_match", match_type]

        self.check_required_columns(data, required_columns)

        # Create a confusion matrix
        confusion_matrix = pd.crosstab(data["ref_match"],
                                    data[match_type])

        # Calculate precision, recall, and F1 score
        precision = np.diag(confusion_matrix) / np.sum(confusion_matrix,
                                                    axis=0)
        recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
        f1_score = 2 * (precision * recall) / (precision + recall)
        f1_score = np.nanmean(f1_score)

        return f1_score

    def mcnemar_test(self, data_a, data_b, top_k_list=None):
        """
        Run McNemar's test to compare two models on paired predictions.

        For each query, a model is considered correct if ref_match appears
        in its top-k matches.  The 2x2 contingency table is:
            b = A correct, B wrong
            c = A wrong,   B correct
        McNemar statistic: chi2 = (|b - c| - 1)^2 / (b + c)  (with continuity correction)
        Falls back to exact binomial p-value when b+c < 25.

        Parameters
        ----------
        data_a, data_b : pd.DataFrame
            Two result DataFrames aligned on 'query'.  Must share 'query' and
            'ref_match' columns plus match1..matchK columns.
        top_k_list : list[int], optional
            Which k values to test.  Defaults to [1, 3, 5].

        Returns
        -------
        pd.DataFrame with columns: k, b, c, n_discordant, statistic, p_value, method
        """
        try:
            from scipy.stats import binom, chi2 as chi2_dist
        except ImportError:
            raise ImportError(
                "mcnemar_test requires scipy. "
                "Install with: pip install metaharmonizer[eval]"
            ) from None

        if top_k_list is None:
            top_k_list = [1, 3, 5]

        # Align on query
        merged = data_a[["query", "ref_match"]].merge(
            data_b[["query", "ref_match"]],
            on=["query", "ref_match"],
            how="inner"
        )
        queries = merged["query"].values
        ref_matches = merged["ref_match"].values

        def is_correct(df, queries, ref_matches, k):
            sub = df[df["query"].isin(queries)].set_index("query")
            cols = [f"match{i}" for i in range(1, k + 1)]
            # keep only cols that exist
            cols = [c for c in cols if c in sub.columns]
            results = []
            for q, ref in zip(queries, ref_matches):
                row = sub.loc[q, cols] if q in sub.index else pd.Series(dtype=object)
                results.append(ref in row.values)
            return np.array(results)

        records = []
        for k in top_k_list:
            correct_a = is_correct(data_a, queries, ref_matches, k)
            correct_b = is_correct(data_b, queries, ref_matches, k)

            b = int(( correct_a & ~correct_b).sum())   # A right, B wrong
            c = int((~correct_a &  correct_b).sum())   # A wrong, B right
            n = b + c

            if n == 0:
                records.append({"k": k, "b": b, "c": c, "n_discordant": n,
                                 "statistic": np.nan, "p_value": np.nan,
                                 "method": "n/a (no discordant pairs)"})
                continue

            if n < 25:
                # Exact binomial (two-sided)
                p_val = 2 * min(binom.cdf(min(b, c), n, 0.5),
                                1 - binom.cdf(min(b, c) - 1, n, 0.5))
                method = "exact binomial"
                stat = float(min(b, c))
            else:
                # Chi-squared with continuity correction
                stat = (abs(b - c) - 1) ** 2 / n
                p_val = float(chi2_dist.sf(stat, df=1))
                method = "chi2 w/ continuity correction"

            records.append({"k": k, "b": b, "c": c, "n_discordant": n,
                             "statistic": round(stat, 4),
                             "p_value": round(p_val, 6),
                             "method": method})

        return pd.DataFrame(records)

    def calc_mismatches_by_score_range(self,
                                    data,
                                    ranges,
                                    match_type="match1",
                                    score_type="match1_score"):
        """
        Calculate the number of mismatches for each score range.

        Parameters:
        - data (pandas.DataFrame): The input data containing the model predictions and the ground truth labels.
        - ranges (list): A list of score ranges to use for the calculation.

        Returns:
        - mismatches_by_range (dict): A dictionary containing the number of mismatches for each score range.
        """
        required_columns = ["ref_match", match_type, score_type]

        self.check_required_columns(data, required_columns)

        # Make a copy to avoid modifying original data
        data = data.copy()

        # Create a new column for the score range
        data["score_range"] = pd.cut(data[score_type], ranges)

        # Calculate the number of mismatches for each score range
        mismatches_by_range = (
            data[data["ref_match"] != data[match_type]].groupby(
                "score_range").size().to_dict())
        return mismatches_by_range
