"""
Time series analysis for student feedback data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def prepare_time_series(df, date_column='date', freq='M', agg_column=None, agg_func='count'):
    """
    Prepare time series data from feedback dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing feedback data
    date_column : str
        Name of the column containing dates
    freq : str
        Frequency for resampling ('D' for daily, 'W' for weekly, 'M' for monthly, 'Q' for quarterly)
    agg_column : str, optional
        Column to aggregate (if None, counts feedback entries)
    agg_func : str
        Aggregation function ('count', 'mean', 'sum', etc.)
        
    Returns:
    --------
    pandas.DataFrame
        Time series data
    """
    # Ensure date column is datetime
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Set date as index
    df = df.set_index(date_column)
    
    # Resample and aggregate
    if agg_column is None:
        # Count feedback entries
        time_series = df.resample(freq).size()
        time_series = pd.DataFrame(time_series, columns=['count'])
    else:
        # Aggregate specified column
        if agg_func == 'count':
            time_series = df.resample(freq)[agg_column].count()
        elif agg_func == 'mean':
            time_series = df.resample(freq)[agg_column].mean()
        elif agg_func == 'sum':
            time_series = df.resample(freq)[agg_column].sum()
        else:
            raise ValueError(f"Unknown aggregation function: {agg_func}")
        
        time_series = pd.DataFrame(time_series, columns=[agg_column])
    
    # Reset index for easier handling
    time_series = time_series.reset_index()
    
    return time_series

def analyze_sentiment_over_time(df, date_column='date', sentiment_column='true_sentiment', freq='M'):
    """
    Analyze sentiment trends over time.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing feedback data
    date_column : str
        Name of the column containing dates
    sentiment_column : str
        Name of the column containing sentiment labels
    freq : str
        Frequency for resampling ('D' for daily, 'W' for weekly, 'M' for monthly, 'Q' for quarterly)
        
    Returns:
    --------
    pandas.DataFrame
        Time series data with sentiment counts and percentages
    """
    # Ensure date column is datetime
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Group by date and sentiment
    grouped = df.groupby([pd.Grouper(key=date_column, freq=freq), sentiment_column]).size().unstack(fill_value=0)
    
    # Calculate percentages
    total = grouped.sum(axis=1)
    percentages = grouped.div(total, axis=0) * 100
    
    # Rename columns for clarity
    percentages.columns = [f"{col}_pct" for col in percentages.columns]
    
    # Combine counts and percentages
    result = pd.concat([grouped, percentages], axis=1)
    
    # Reset index for easier handling
    result = result.reset_index()
    
    return result

def analyze_categories_over_time(df, date_column='date', category_column='true_category', freq='M'):
    """
    Analyze category trends over time.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing feedback data
    date_column : str
        Name of the column containing dates
    category_column : str
        Name of the column containing category labels
    freq : str
        Frequency for resampling ('D' for daily, 'W' for weekly, 'M' for monthly, 'Q' for quarterly)
        
    Returns:
    --------
    pandas.DataFrame
        Time series data with category counts
    """
    # Ensure date column is datetime
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Group by date and category
    grouped = df.groupby([pd.Grouper(key=date_column, freq=freq), category_column]).size().unstack(fill_value=0)
    
    # Reset index for easier handling
    result = grouped.reset_index()
    
    return result

def analyze_ratings_over_time(df, date_column='date', rating_column='rating', freq='M'):
    """
    Analyze rating trends over time.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing feedback data
    date_column : str
        Name of the column containing dates
    rating_column : str
        Name of the column containing ratings
    freq : str
        Frequency for resampling ('D' for daily, 'W' for weekly, 'M' for monthly, 'Q' for quarterly)
        
    Returns:
    --------
    pandas.DataFrame
        Time series data with average ratings
    """
    # Ensure date column is datetime
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Group by date and calculate average rating
    grouped = df.groupby(pd.Grouper(key=date_column, freq=freq))[rating_column].agg(['mean', 'count', 'std'])
    
    # Reset index for easier handling
    result = grouped.reset_index()
    
    return result

def detect_anomalies(time_series, column, window=3, threshold=2):
    """
    Detect anomalies in time series data using rolling statistics.
    
    Parameters:
    -----------
    time_series : pandas.DataFrame
        Time series data
    column : str
        Column to analyze for anomalies
    window : int
        Window size for rolling statistics
    threshold : float
        Number of standard deviations to consider as anomaly
        
    Returns:
    --------
    pandas.DataFrame
        Time series with anomaly indicators
    """
    result = time_series.copy()
    
    # Calculate rolling mean and standard deviation
    rolling_mean = result[column].rolling(window=window).mean()
    rolling_std = result[column].rolling(window=window).std()
    
    # Calculate upper and lower bounds
    upper_bound = rolling_mean + (rolling_std * threshold)
    lower_bound = rolling_mean - (rolling_std * threshold)
    
    # Identify anomalies
    result['anomaly'] = 0
    result.loc[result[column] > upper_bound, 'anomaly'] = 1
    result.loc[result[column] < lower_bound, 'anomaly'] = -1
    
    # Add bounds for visualization
    result['upper_bound'] = upper_bound
    result['lower_bound'] = lower_bound
    result['rolling_mean'] = rolling_mean
    
    return result

def plot_time_series(time_series, date_column, value_column, title=None, figsize=(12, 6)):
    """
    Plot time series data.
    
    Parameters:
    -----------
    time_series : pandas.DataFrame
        Time series data
    date_column : str
        Name of the column containing dates
    value_column : str
        Name of the column to plot
    title : str, optional
        Plot title
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(time_series[date_column], time_series[value_column], marker='o')
    
    # Add labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel(value_column)
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'{value_column} Over Time')
    
    # Format x-axis
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def plot_sentiment_trends(sentiment_time_series, date_column='date', figsize=(12, 8)):
    """
    Plot sentiment trends over time.
    
    Parameters:
    -----------
    sentiment_time_series : pandas.DataFrame
        Time series data with sentiment counts and percentages
    date_column : str
        Name of the column containing dates
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Get sentiment columns (excluding date and percentage columns)
    sentiment_columns = [col for col in sentiment_time_series.columns 
                        if col != date_column and not col.endswith('_pct')]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot counts
    for col in sentiment_columns:
        ax1.plot(sentiment_time_series[date_column], sentiment_time_series[col], marker='o', label=col)
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Count')
    ax1.set_title('Sentiment Counts Over Time')
    ax1.legend()
    
    # Plot percentages
    for col in sentiment_columns:
        pct_col = f"{col}_pct"
        if pct_col in sentiment_time_series.columns:
            ax2.plot(sentiment_time_series[date_column], sentiment_time_series[pct_col], marker='o', label=col)
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Sentiment Percentages Over Time')
    ax2.legend()
    
    # Format x-axis
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def plot_category_trends(category_time_series, date_column='date', figsize=(14, 10)):
    """
    Plot category trends over time.
    
    Parameters:
    -----------
    category_time_series : pandas.DataFrame
        Time series data with category counts
    date_column : str
        Name of the column containing dates
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Get category columns (excluding date column)
    category_columns = [col for col in category_time_series.columns if col != date_column]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot stacked area chart
    category_time_series.plot(x=date_column, y=category_columns, kind='area', stacked=True, ax=ax)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Count')
    ax.set_title('Category Trends Over Time')
    
    # Format x-axis
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def plot_rating_trends(rating_time_series, date_column='date', figsize=(12, 8)):
    """
    Plot rating trends over time.
    
    Parameters:
    -----------
    rating_time_series : pandas.DataFrame
        Time series data with average ratings
    date_column : str
        Name of the column containing dates
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Plot average rating
    ax1.plot(rating_time_series[date_column], rating_time_series['mean'], marker='o', color='blue')
    
    # Add error bars
    if 'std' in rating_time_series.columns:
        ax1.fill_between(
            rating_time_series[date_column],
            rating_time_series['mean'] - rating_time_series['std'],
            rating_time_series['mean'] + rating_time_series['std'],
            alpha=0.2, color='blue'
        )
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Average Rating')
    ax1.set_title('Average Rating Over Time')
    
    # Plot feedback count
    ax2.bar(rating_time_series[date_column], rating_time_series['count'], color='green')
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Feedback Count')
    ax2.set_title('Feedback Volume Over Time')
    
    # Format x-axis
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def plot_anomalies(anomaly_time_series, date_column, value_column, title=None, figsize=(12, 6)):
    """
    Plot time series with anomalies highlighted.
    
    Parameters:
    -----------
    anomaly_time_series : pandas.DataFrame
        Time series data with anomaly indicators
    date_column : str
        Name of the column containing dates
    value_column : str
        Name of the column to plot
    title : str, optional
        Plot title
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the time series
    ax.plot(anomaly_time_series[date_column], anomaly_time_series[value_column], marker='o', label=value_column)
    
    # Plot rolling mean
    if 'rolling_mean' in anomaly_time_series.columns:
        ax.plot(anomaly_time_series[date_column], anomaly_time_series['rolling_mean'], 
               color='green', linestyle='--', label='Rolling Mean')
    
    # Plot bounds
    if 'upper_bound' in anomaly_time_series.columns and 'lower_bound' in anomaly_time_series.columns:
        ax.fill_between(
            anomaly_time_series[date_column],
            anomaly_time_series['lower_bound'],
            anomaly_time_series['upper_bound'],
            alpha=0.2, color='green'
        )
    
    # Highlight anomalies
    if 'anomaly' in anomaly_time_series.columns:
        # Positive anomalies
        positive_anomalies = anomaly_time_series[anomaly_time_series['anomaly'] == 1]
        if not positive_anomalies.empty:
            ax.scatter(positive_anomalies[date_column], positive_anomalies[value_column], 
                      color='red', s=100, label='Positive Anomaly')
        
        # Negative anomalies
        negative_anomalies = anomaly_time_series[anomaly_time_series['anomaly'] == -1]
        if not negative_anomalies.empty:
            ax.scatter(negative_anomalies[date_column], negative_anomalies[value_column], 
                      color='purple', s=100, label='Negative Anomaly')
    
    # Add labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel(value_column)
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'{value_column} Over Time with Anomalies')
    
    ax.legend()
    
    # Format x-axis
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig
