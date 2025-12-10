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

    def calc_accuracy(self, data, mode="t5", custom_top_k=None):
        """
        Calculate the accuracy of the model predictions for multiple top-k values.

        Parameters:
        - data (pandas.DataFrame): The input data containing the model predictions and the ground truth labels.
        - mode (str): Evaluation mode. Options:
            - "t5": Standard top-1, 3, 5 evaluation (default)
            - "rerank": Evaluate reranking candidate quality at top-1, 5, 20, 30, 50
        - custom_top_k (list, optional): Custom list of top-k values. Overrides mode if provided.

        Returns:
        - accuracy_df (pandas.DataFrame): A DataFrame containing the accuracy levels and their corresponding accuracies.
        """
        # Determine top-k list based on mode
        if custom_top_k is not None:
            top_k_list = custom_top_k
        elif mode == "t5":
            top_k_list = [1, 3, 5]
        elif mode == "rerank":
            top_k_list = [1, 5, 20, 30, 50]
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Choose 't5' or 'rerank', or provide custom_top_k."
            )

        # Determine maximum k needed
        max_k = max(top_k_list)

        # Build required columns list
        required_columns = ["curated_ontology"]
        required_columns.extend([f"top{i}_match" for i in range(1, max_k + 1)])

        # Check if all required columns exist
        self.check_required_columns(data, required_columns)

        # Calculate accuracy for each top-k
        accuracy_results = []

        for k in top_k_list:
            # Get the column names for top-k matches
            top_k_columns = [f"top{i}_match" for i in range(1, k + 1)]

            # Calculate if curated_ontology is in top-k matches
            accuracy_col = f"top{k}_accuracy"
            data[accuracy_col] = data.apply(lambda row: row["curated_ontology"]
                                            in row[top_k_columns].values,
                                            axis=1)

            # Calculate percentage accuracy
            accuracy_pct = data[accuracy_col].mean() * 100

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
        - match_type (str): The type of match to use for the confusion matrix. Can be 'top1_match', 'top3_match', or 'top5_match'.

        Returns:
        - confusion_matrix_df (pandas.DataFrame): A DataFrame containing the confusion matrix values.
        """
        required_columns = ["curated_ontology", match_type]

        self.check_required_columns(data, required_columns)
        # Create a confusion matrix
        confusion_matrix = pd.crosstab(data["curated_ontology"],
                                       data[match_type])
        confusion_matrix_df = pd.DataFrame(confusion_matrix)

        return confusion_matrix_df

    def calc_f1_score(self, data, match_type):
        """
        Calculate the F1 score for the model predictions.

        Parameters:
        - data (pandas.DataFrame): The input data containing the model predictions and the ground truth labels.
        - match_type (str): The type of match to use for the F1 score calculation. Can be 'top1_match', 'top3_match', or 'top5_match'.

        Returns:
        - f1_score (float): The F1 score for the model predictions.
        """
        required_columns = ["curated_ontology", match_type]

        self.check_required_columns(data, required_columns)

        # Create a confusion matrix
        confusion_matrix = pd.crosstab(data["curated_ontology"],
                                       data[match_type])

        # Calculate precision, recall, and F1 score
        precision = np.diag(confusion_matrix) / np.sum(confusion_matrix,
                                                       axis=0)
        recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
        f1_score = 2 * (precision * recall) / (precision + recall)
        f1_score = np.nanmean(f1_score)

        return f1_score

    def calc_mismatches_by_score_range(self,
                                       data,
                                       ranges,
                                       match_type="top1_match",
                                       score_type="top1_score"):
        """
        Calculate the number of mismatches for each score range.

        Parameters:
        - data (pandas.DataFrame): The input data containing the model predictions and the ground truth labels.
        - ranges (list): A list of score ranges to use for the calculation.

        Returns:
        - mismatches_by_range (dict): A dictionary containing the number of mismatches for each score range.
        """
        required_columns = ["curated_ontology", match_type, score_type]

        self.check_required_columns(data, required_columns)

        # Create a new column for the score range
        data["score_range"] = pd.cut(data[score_type], ranges)

        # Calculate the number of mismatches for each score range
        mismatches_by_range = (
            data[data["curated_ontology"] != data[match_type]].groupby(
                "score_range").size().to_dict())
        return mismatches_by_range
