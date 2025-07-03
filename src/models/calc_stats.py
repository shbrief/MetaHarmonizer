import pandas as pd
import numpy as np

# import statsmodels.api as sm


class CalcStats:

    def __init__(self) -> None:
        pass
        # self.required_columns = ['curated_ontology', 'top1_match', 'top2_match', 'top3_match', 'top4_match', 'top5_match']

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

    def calc_accuracy(self, data):
        """
        Calculate the accuracy of the model predictions.

        Parameters:
        - data (pandas.DataFrame): The input data containing the model predictions and the ground truth labels.

        Returns:
        - accuracy_df (pandas.DataFrame): A DataFrame containing the accuracy levels and their corresponding accuracies.

        """
        required_columns = [
            "curated_ontology",
            "top1_match",
            "top2_match",
            "top3_match",
            "top4_match",
            "top5_match",
        ]

        # Check if all required columns exist in the DataFrame

        self.check_required_columns(data, required_columns)

        # Calculate accuracy for Top 1, Top 3, and Top 5 matches
        data["top1_accuracy"] = data.apply(
            lambda row: row["curated_ontology"] == row["top1_match"], axis=1)
        data["top3_accuracy"] = data.apply(
            lambda row: row["curated_ontology"] in row[
                ["top1_match", "top2_match", "top3_match"]].values,
            axis=1,
        )
        data["top5_accuracy"] = data.apply(
            lambda row: row["curated_ontology"] in row[[
                "top1_match", "top2_match", "top3_match", "top4_match",
                "top5_match"
            ]].values,
            axis=1,
        )

        # Calculate percentage accuracies
        top1_accuracy = data["top1_accuracy"].mean() * 100
        top3_accuracy = data["top3_accuracy"].mean() * 100
        top5_accuracy = data["top5_accuracy"].mean() * 100

        # Prepare data for plotting
        accuracy_data = {
            "Accuracy Level":
            ["Top 1 Match", "Top 3 Matches", "Top 5 Matches"],
            "Accuracy": [top1_accuracy, top3_accuracy, top5_accuracy],
        }

        accuracy_df = pd.DataFrame(accuracy_data)

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
        data["score_range"] = pd.cut(data["top1_score"], ranges)

        # Calculate the number of mismatches for each score range
        mismatches_by_range = (
            data[data["curated_ontology"] != data["top1_match"]].groupby(
                "score_range").size().to_dict())
        return mismatches_by_range
