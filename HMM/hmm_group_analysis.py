import pandas as pd

def create_hypothesis_testing_dataframe(group_data: dict, group_labels: list, metrics: list, k: int):
    """
    Args:
        group_data   : dict
        group_labels : list
        metrics      : list
        k            : int
    Returns:
        ht_df        : pandas DataFrame
    """
    group_dfs = []

    for label in group_labels:
        group_df = pd.DataFrame({
            f'{metric}_state{state + 1}': list(group_data[label][metric][:, state]) for metric in metrics for state in range(k)
        })
        group_df['group'] = label
    
        group_dfs.append(group_df)
    
    ht_df = pd.concat(group_dfs, axis=0, ignore_index=True)
    return ht_df

def separate_trans_probs(group_data: dict, group_labels: list, k: int):
    """
    Args:
        group_data   : dict
        group_labels : list
        k            : int
    Returns:
        tp_df        : pandas DataFrame
    """
    group_dfs = []
    for label in list(group_labels):
        group_df = pd.DataFrame({
            f'tp_{i + 1}to{j + 1}': list(group_data[label]['trans_matrices'][:, i, j]) for i in range(k) for j in range(k)
        })
        group_df['group'] = label
        
        group_dfs.append(group_df)
        
    tp_df = pd.concat(group_dfs, axis=0, ignore_index=True)
    return tp_df