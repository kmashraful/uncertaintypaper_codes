import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_comparison_boxplot(input_df, metric, output_path, y_limits=[0.9, 1.0]):
    rename_dict = {
        "RandomForest": "Random\nForests",
        "SVC": "Support Vector\nMachine",
        "KNN": "k-Nearest\nNeighbors",
        "LogisticRegression": "Logistic\nRegression",
        "XGBoost": "XGBoost"
    }
    
    input_df = input_df.copy()
    input_df['Classifier_Type'] = input_df['Classifier_Type'].replace(rename_dict)
    
    classifier_order = sorted(input_df['Classifier_Type'].unique())
    
    sns.set_style("white")
    palette = sns.color_palette("RdYlGn", n_colors=len(classifier_order))
    
    plt.figure(figsize=(6, 4))
    sns.boxplot(
        x='Classifier_Type',
        y=metric,
        data=input_df,
        order=classifier_order,
        palette=palette,
        boxprops=dict(edgecolor='black')
    )
    
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('')
    plt.ylim(y_limits)
    
    plt.tight_layout()
    # plt.show()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return None

if __name__ == "__main__":
    
    # Define the folder wher outputs are stored
    output_folder = "C:/Users/johnb/Documents/git/masters_project/stacked_generalization/outputs"
    
    # Make the output figure folder
    figure_folder = f"{output_folder}/figure_folder"
    if not os.path.exists(figure_folder):
        os.mkdir(figure_folder)
    
    # Read in the cross validation results for the base-learners
    base_learner_df = pd.read_csv(f"{output_folder}/baselearner_cv_accuracy.csv")
    base_learner_df.rename(columns={'F1_Score': 'F1-Score'}, inplace=True)

    # Read in the cross-validation results for the stacking models
    stacking_df = pd.read_csv(f"{output_folder}/stacking_cv_accuracy.csv")
    stacking_df.rename(columns={'F1_Score': 'F1-Score'}, inplace=True)

    # Generate the plots
    metrics = ['Accuracy', 'F1-Score', 'AUC', 'Precision', 'Recall']
    for model_set in ["baselearner"]:
        for metric in metrics:
            
            # Get the dataframe to plot
            if model_set == "baselearner":
                df_to_plot = base_learner_df
            else:
                df_to_plot = stacking_df
                
            # Define the output path name
            output_path = f"{figure_folder}/{model_set}_{metric}.PNG"
            
            # Plot the dataset
            plot_comparison_boxplot(base_learner_df, metric, output_path)
            
    
            
    
    
