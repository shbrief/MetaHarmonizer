import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd 


class PlotModelPerformance:
    def __init__(self) -> None:
        pass

    def create_accuracy_bar_plot(self, accuracy_df, title):
        # Plotting
        plt.figure(figsize=(8, 5))
        plt.bar(accuracy_df['Accuracy Level'], accuracy_df['Accuracy'], color=['blue', 'orange', 'green'])
        plt.xlabel('Match Level')
        plt.ylabel('Accuracy (%)')
        plt.title(title)
        plt.ylim(0, 100)
        plt.show()
        return 

    def create_confusion_matrix_plot(self, confusion_matrix, title):
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(title)
        plt.show()
        return
    
    @abstractmethod
    def create_similarity_heatmap_plot(self, cosine_sim_df, title):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cosine_sim_df, annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Corpus')
        plt.ylabel('Query')
        plt.title(title)
        plt.show()
        return
    
    def compare_model_performances(self, top_k, models, categories, accuracies):
        """
        Function to compare the performance of different models based on the accuracy scores.

        Parameters:
        - top_k (list): List of integers representing the top K matches.
        - models (list): List of strings representing the model names.
        - categories (list): List of strings representing the ontology categories.
        - accuracies (dict): Dictionary containing the accuracy scores for each model and category.
                             The keys are the model names, and the values are dictionaries where the keys are the
                             ontology categories and the values are lists of accuracy scores corresponding to each
                             top K match.

        Returns:
        None

        """
        # Saving the plot with high resolution
        colors = {
            'Bodysite': 'blue',
            'Disease': 'orange',
            'Treatment': 'green'
        }

        # Adjusting plot layout for enhanced legend readability and high-resolution output
        fig, ax = plt.subplots(figsize=(10, 8))  # Increased height

        # Lines for each category
        line_styles = ['-', '--', '-.', ':']
        line_style_index = 0
        for model, data in accuracies.items():
            if line_style_index >= len(line_styles):
                line_style_index = 0
            line_style = line_styles[line_style_index]
            line_style_index += 1
            for category, acc in data.items():
                ax.plot(top_k, acc, line_style, color=colors[category], label=f'{category} ({model})')

        # Creating custom legends with enhanced font size and adjusted positions
        # Legend for models
        lines = [plt.Line2D([0], [0], color='black', linestyle=line_styles[i], label=model) for i, model in enumerate(models)]
        model_legend = ax.legend(handles=lines, title='Model', loc='lower right', bbox_to_anchor=(1, 0), fontsize=10, title_fontsize=12)
        plt.gca().add_artist(model_legend)

        # Legend for ontology categories
        category_lines = [plt.Line2D([0], [0], color=color, label=cat) for cat, color in colors.items()]
        category_legend = ax.legend(handles=category_lines, title='Ontology Category', loc='lower right', bbox_to_anchor=(1, 0.2), fontsize=10, title_fontsize=12)

        # Setting the x-axis to show integer values and renaming labels
        ax.set_xticks(top_k)
        ax.set_xticklabels([f'Top{k}' for k in top_k])
        ax.set_xlim(0.5, len(top_k) + 0.5)  # Reducing the white space by adjusting xlim

        # Adjusting Y-axis
        ax.set_ylim(0, 100)

        # Labels and title
        ax.set_xlabel("Top K Match", fontsize=14)
        ax.set_ylabel("Accuracy (%)", fontsize=14)
        ax.set_title("Accuracy Comparison", fontsize=16)
        ax.grid(True)
    
