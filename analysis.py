import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def display_importances(feature_importance_array, feature_names, n):
    feature_importance_array = np.mean(feature_importance_array, axis=0)
    feature_importance_df_ = pd.DataFrame()
    feature_importance_df_["feature"] = feature_names
    feature_importance_df_["importance"] = np.array(feature_importance_array)

    cols = feature_importance_df_.sort_values(
        by="importance", ascending=False)[:n].index
    best_features = feature_importance_df_.loc[cols]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(
        by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')
    return feature_importance_df_
