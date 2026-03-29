import pandas as pd


def format_cv_results(df):

    # Group by Classifier_Type and compute mean, std
    grouped = df.groupby("Classifier_Type").agg(["mean", "std"])

    # Round to 2 decimal places
    grouped = grouped.round(3)

    # Build a new DataFrame with columns in "mean +/- std" format
    formatted_df = pd.DataFrame(index=grouped.index)
    for col in grouped.columns.levels[0]:
        if col != "Classifier_Type":
            mean_col = (col, "mean")
            std_col = (col, "std")
            formatted_df[col] = grouped[mean_col].astype(str) + " $\pm$ " + grouped[std_col].astype(str)

    # Move index to column
    formatted_df.reset_index(inplace=True)

    return formatted_df


if __name__ == "__main__":
    
    # Define the folder wher outputs are stored
    output_folder = "C:/Users/johnb/Documents/git/masters_project/stacked_generalization/outputs"
    
    ###########################################################################
    ### STEP 2. COMPUTE THE ACCURACY OF THE STACKING MODELS 
    ###########################################################################
    
    # Read in the stacking model CV accuracy
    stacking_metrics_df = pd.read_csv(f"{output_folder}/model_cv_accuracy.csv")
    
    # Subset the features that are needed
    stacking_metrics_df = stacking_metrics_df[[
        'Classifier_Type', 'Accuracy', 'F1_Score', 'AUC', 
        'Precision', 'Recall', "logloss"
        ]]

    # Compute the mean/std of each column
    stacking_summary_df = format_cv_results(stacking_metrics_df)
    stacking_summary_df.to_csv(f"{output_folder}/formatted_cv_accuracy.csv")

