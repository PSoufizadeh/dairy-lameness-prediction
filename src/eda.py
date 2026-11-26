"""Exploratory summaries; outputs must remain local."""
def missingness(frame):
    return frame.isna().sum()


def summarize_by_score(frame, score, measurement):
    return frame.groupby(score)[measurement].agg(['count', 'mean', 'median', 'std'])


def score_transition_table(frame, previous_score, current_score):
    import pandas as pd
    return pd.crosstab(frame[previous_score], frame[current_score], normalize='columns')
