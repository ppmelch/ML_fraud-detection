def compute_scale_pos_weight(y_train):
    '''
    Compute the scale_pos_weight for XGBoost based on the training labels.
    This is calculated as the ratio of the number of negative samples to positive samples.
    Parameters
    ----------
    y_train : pd.Series
        Training labels (binary classification).
        Returns
        -------
        float
            The computed scale_pos_weight value.
    '''
    
    return (
        y_train.value_counts()[0] /
        y_train.value_counts()[1]
    )