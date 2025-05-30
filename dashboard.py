"""
Streamlit dashboard for Student Feedback Analysis.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from datetime import datetime, timedelta
import uuid
import json

# Import project modules
from src.utils.data_generator import generate_sample_feedback
from src.preprocessing.text_processor import TextProcessor, extract_features
from src.models.supervised_models import SentimentClassifier, FeedbackCategorizer
from src.models.unsupervised_models import TopicModeler, FeedbackClusterer, DimensionalityReducer
from src.evaluation.metrics import evaluate_classification, evaluate_clustering
from src.visualization.visualizer import plot_wordcloud, plot_topic_wordcloud, plot_cluster_visualization
from src.analysis.time_series import (analyze_sentiment_over_time, analyze_categories_over_time,
                                    analyze_ratings_over_time, detect_anomalies)
from src.data.feedback_collector import FeedbackCollector
from src.auth.user_auth import UserAuth

# Set page configuration
st.set_page_config(
    page_title="Student Feedback Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize authentication
def init_auth():
    """Initialize authentication system."""
    auth = UserAuth()

    # Create admin user if none exists
    admin_result = auth.create_admin_if_none()
    if admin_result['status'] == 'success':
        st.sidebar.warning(
            f"Default admin created! Username: {admin_result['username']}, "
            f"Password: {admin_result['password']}. Please change this password immediately."
        )

    return auth

# Session management functions
def get_session_state():
    """Get or initialize session state."""
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'auth_status' not in st.session_state:
        st.session_state.auth_status = None
    if 'auth_message' not in st.session_state:
        st.session_state.auth_message = None
    if 'token' not in st.session_state:
        st.session_state.token = None

def login_user(auth, username, password):
    """Login a user and set session state."""
    result = auth.login(username, password)

    if result['status'] == 'success':
        st.session_state.user = {
            'username': result['username'],
            'role': result['role'],
            'full_name': result['full_name']
        }
        st.session_state.token = result['token']
        st.session_state.auth_status = 'success'
        st.session_state.auth_message = result['message']
    else:
        st.session_state.auth_status = 'error'
        st.session_state.auth_message = result['message']

def logout_user(auth):
    """Logout a user and clear session state."""
    if st.session_state.token:
        auth.logout(st.session_state.token)

    st.session_state.user = None
    st.session_state.token = None
    st.session_state.auth_status = None
    st.session_state.auth_message = None

def check_authentication(auth):
    """Check if user is authenticated."""
    if st.session_state.token:
        result = auth.validate_session(st.session_state.token)
        if result['status'] == 'success':
            return True
    return False

def get_user_role():
    """Get the role of the current user."""
    if st.session_state.user:
        return st.session_state.user['role']
    return None

# Define functions for the dashboard
def load_data():
    """Load or generate sample data."""
    data_path = 'data/sample_feedback.csv'
    try:
        if not os.path.exists(data_path):
            st.info("Generating sample feedback data...")
            df = generate_sample_feedback(n_samples=1000, output_path=data_path)
            if df is None:
                st.error(f"Failed to generate sample data.")
                return pd.DataFrame()  # Return empty DataFrame instead of None
        else:
            df = pd.read_csv(data_path)
            if df.empty:
                st.warning(f"The data file {data_path} exists but is empty.")

        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error

def preprocess_data(df):
    """Preprocess the text data."""
    if 'processed_text' not in df.columns:
        with st.spinner("Preprocessing text data..."):
            text_processor = TextProcessor(
                remove_stopwords=True,
                remove_punctuation=True,
                lemmatize=True,
                stem=False,
                lowercase=True
            )
            df = text_processor.preprocess_dataframe(df, 'feedback_text')

    return df

def extract_text_features(df):
    """Extract features from text data."""
    with st.spinner("Extracting features..."):
        feature_matrix, vectorizer = extract_features(
            df['processed_text'],
            method='tfidf',
            max_features=5000,
            ngram_range=(1, 2)
        )

    return feature_matrix, vectorizer

def train_sentiment_model(X_train, y_train):
    """Train sentiment classification model."""
    with st.spinner("Training sentiment classifier..."):
        model = SentimentClassifier(model_type='logistic_regression')
        model.train(X_train, y_train)

    return model

def train_category_model(X_train, y_train):
    """Train category classification model."""
    with st.spinner("Training category classifier..."):
        model = FeedbackCategorizer(model_type='random_forest')
        model.train(X_train, y_train)

    return model

def perform_topic_modeling(texts, n_topics=5):
    """Perform topic modeling."""
    with st.spinner("Performing topic modeling..."):
        topic_modeler = TopicModeler(method='lda', n_topics=n_topics)
        topic_modeler.fit(texts)

    return topic_modeler

def perform_clustering(features, n_clusters=5):
    """Perform clustering."""
    with st.spinner("Performing clustering..."):
        # Reduce dimensionality for clustering
        reducer = DimensionalityReducer(method='svd', n_components=50)
        reduced_features = reducer.fit_transform(features)

        clusterer = FeedbackClusterer(method='kmeans', n_clusters=n_clusters)
        clusterer.fit(reduced_features)

    return clusterer, reduced_features

# Main dashboard
def main():
    """Main dashboard function."""
    # Initialize authentication
    auth = init_auth()

    # Initialize session state
    get_session_state()

    # Sidebar
    st.sidebar.title("Student Feedback Analysis")
    st.sidebar.image("https://img.icons8.com/color/96/000000/student-center.png", width=100)

    # Authentication UI in sidebar
    if not check_authentication(auth):
        st.sidebar.subheader("Login")
        with st.sidebar.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")

            if submit_button:
                login_user(auth, username, password)

        # Show registration option
        st.sidebar.markdown("---")
        if st.sidebar.button("Register New Account"):
            st.session_state.show_register = True

        # Display authentication messages
        if st.session_state.auth_status == 'error':
            st.sidebar.error(st.session_state.auth_message)

        # Registration form
        if st.session_state.get('show_register', False):
            st.sidebar.subheader("Register")
            with st.sidebar.form("register_form"):
                new_username = st.text_input("New Username")
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                email = st.text_input("Email")
                full_name = st.text_input("Full Name")
                role = st.selectbox("Role", ["student", "instructor"])

                register_button = st.form_submit_button("Register")

                if register_button:
                    if new_password != confirm_password:
                        st.sidebar.error("Passwords do not match")
                    elif not new_username or not new_password or not email:
                        st.sidebar.error("Please fill in all required fields")
                    else:
                        result = auth.register_user(
                            username=new_username,
                            password=new_password,
                            email=email,
                            role=role,
                            full_name=full_name
                        )

                        if result['status'] == 'success':
                            st.sidebar.success(result['message'])
                            st.session_state.show_register = False
                        else:
                            st.sidebar.error(result['message'])

        # Show limited content for non-authenticated users
        st.title("Student Feedback Analysis System")
        st.write("Please login to access the dashboard.")

        # Show sample features
        st.subheader("Features Available After Login")
        st.write("""
        - Interactive data exploration
        - Sentiment analysis of feedback
        - Feedback categorization
        - Topic modeling to discover hidden themes
        - Clustering similar feedback
        - Time series analysis and trend tracking
        - Real-time feedback collection
        """)

        return

    # User is authenticated - show logout button and user info
    st.sidebar.subheader(f"Welcome, {st.session_state.user['full_name'] or st.session_state.user['username']}")
    st.sidebar.write(f"Role: {st.session_state.user['role'].capitalize()}")

    if st.sidebar.button("Logout"):
        logout_user(auth)
        try:
            st.rerun()  # For newer versions of Streamlit
        except AttributeError:
            st.experimental_rerun()  # For older versions of Streamlit

    # Role-based page access
    role = get_user_role()

    available_pages = ["Home"]

    if role == 'admin':
        available_pages.extend([
            "Data Exploration", "Sentiment Analysis", "Feedback Categories",
            "Topic Modeling", "Clustering", "Time Series Analysis", "Combined Analysis",
            "Feedback Collection", "User Management"
        ])
    elif role == 'instructor':
        available_pages.extend([
            "Data Exploration", "Sentiment Analysis", "Feedback Categories",
            "Topic Modeling", "Time Series Analysis", "Feedback Collection"
        ])
    elif role == 'student':
        available_pages.extend(["Feedback Collection"])

    page = st.sidebar.selectbox("Choose a page", available_pages)

    # Load data
    df = load_data()

    # Home page
    if page == "Home":
        st.title("Student Feedback Analysis Dashboard")
        st.write("""
        This dashboard provides interactive visualizations and insights from student feedback data
        using both supervised and unsupervised machine learning techniques.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Dataset Overview")
            st.write(f"Total feedback samples: {len(df)}")
            st.write(f"Subjects: {df['subject'].nunique()}")
            st.write(f"Sentiment distribution: {df['true_sentiment'].value_counts().to_dict()}")
            st.write(f"Category distribution: {df['true_category'].value_counts().to_dict()}")

        with col2:
            st.subheader("Sample Feedback")
            sample_df = df[['feedback_text', 'true_sentiment', 'true_category']].sample(5)
            st.dataframe(sample_df)

        st.subheader("Navigate the Dashboard")
        st.write("""
        Use the sidebar to navigate through different analysis pages:
        - **Data Exploration**: Explore the raw data and distributions
        - **Sentiment Analysis**: Analyze sentiment in feedback
        - **Feedback Categories**: Explore feedback categories
        - **Topic Modeling**: Discover hidden themes in feedback
        - **Clustering**: Group similar feedback together
        - **Time Series Analysis**: Track feedback trends over time
        - **Combined Analysis**: Combine supervised and unsupervised insights
        - **Feedback Collection**: Submit and manage new feedback
        """)

    # Data Exploration page
    elif page == "Data Exploration":
        st.title("Data Exploration")

        # Preprocess data
        df = preprocess_data(df)

        # Display data
        st.subheader("Raw Data")
        st.dataframe(df.head(10))

        # Display statistics
        st.subheader("Data Statistics")
        st.write(df.describe())

        # Visualizations
        st.subheader("Visualizations")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Sentiment Distribution")
            fig = px.pie(df, names='true_sentiment', title='Sentiment Distribution')
            st.plotly_chart(fig)

        with col2:
            st.write("Rating Distribution")
            fig = px.histogram(df, x='rating', title='Rating Distribution',
                              color='true_sentiment', barmode='group')
            st.plotly_chart(fig)

        st.write("Category Distribution")
        fig = px.bar(df['true_category'].value_counts().reset_index(),
                    x='index', y='true_category', title='Category Distribution')
        st.plotly_chart(fig)

        st.write("Subject Distribution")
        fig = px.bar(df['subject'].value_counts().reset_index(),
                    x='index', y='subject', title='Subject Distribution')
        st.plotly_chart(fig)

        # Word cloud
        st.subheader("Word Cloud of Feedback")

        # Create word cloud
        wc_text = " ".join(df['processed_text'].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wc_text)

        # Display word cloud
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    # Sentiment Analysis page
    elif page == "Sentiment Analysis":
        st.title("Sentiment Analysis")

        # Preprocess data
        df = preprocess_data(df)

        # Extract features
        feature_matrix, vectorizer = extract_text_features(df)

        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            feature_matrix, df['true_sentiment'], test_size=0.3, random_state=42
        )

        # Train model
        sentiment_model = train_sentiment_model(X_train, y_train)

        # Evaluate model
        results = sentiment_model.evaluate(X_test, y_test)

        # Display results
        st.subheader("Model Performance")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Accuracy", f"{results['accuracy']:.4f}")

        with col2:
            st.metric("F1 Score", f"{results['f1_score']:.4f}")

        # Display classification report
        st.subheader("Classification Report")
        report_df = pd.DataFrame(results['classification_report']).transpose()
        st.dataframe(report_df)

        # Confusion matrix
        st.subheader("Confusion Matrix")
        y_pred = sentiment_model.predict(X_test)
        cm = evaluate_classification(y_test, y_pred)['confusion_matrix']

        fig = px.imshow(cm,
                       x=['Negative', 'Neutral', 'Positive'],
                       y=['Negative', 'Neutral', 'Positive'],
                       text_auto=True,
                       title='Confusion Matrix',
                       color_continuous_scale='Blues')
        st.plotly_chart(fig)

        # Sentiment by subject
        st.subheader("Sentiment by Subject")
        sentiment_by_subject = pd.crosstab(df['subject'], df['true_sentiment'])
        sentiment_by_subject_pct = sentiment_by_subject.div(sentiment_by_subject.sum(axis=1), axis=0)

        fig = px.bar(sentiment_by_subject_pct.reset_index().melt(id_vars='subject'),
                    x='subject', y='value', color='true_sentiment',
                    title='Sentiment Distribution by Subject',
                    labels={'value': 'Percentage', 'subject': 'Subject'})
        st.plotly_chart(fig)

        # Sentiment prediction
        st.subheader("Predict Sentiment")
        user_input = st.text_area("Enter feedback text to predict sentiment:",
                                 "I really enjoyed the course. The teaching was excellent.")

        if st.button("Predict"):
            # Preprocess input
            text_processor = TextProcessor()
            processed_input = text_processor.preprocess(user_input)

            # Transform to features
            input_features = vectorizer.transform([processed_input])

            # Predict
            prediction = sentiment_model.predict(input_features)[0]

            # Display result
            st.success(f"Predicted sentiment: {prediction}")

    # Feedback Categories page
    elif page == "Feedback Categories":
        st.title("Feedback Categories")

        # Preprocess data
        df = preprocess_data(df)

        # Extract features
        feature_matrix, vectorizer = extract_text_features(df)

        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            feature_matrix, df['true_category'], test_size=0.3, random_state=42
        )

        # Train model
        category_model = train_category_model(X_train, y_train)

        # Evaluate model
        results = category_model.evaluate(X_test, y_test)

        # Display results
        st.subheader("Model Performance")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Accuracy", f"{results['accuracy']:.4f}")

        with col2:
            st.metric("F1 Score", f"{results['f1_score']:.4f}")

        # Display classification report
        st.subheader("Classification Report")
        report_df = pd.DataFrame(results['classification_report']).transpose()
        st.dataframe(report_df)

        # Category distribution
        st.subheader("Category Distribution")
        fig = px.pie(df, names='true_category', title='Category Distribution')
        st.plotly_chart(fig)

        # Category by subject
        st.subheader("Categories by Subject")
        category_by_subject = pd.crosstab(df['subject'], df['true_category'])

        fig = px.bar(category_by_subject.reset_index().melt(id_vars='subject'),
                    x='subject', y='value', color='true_category',
                    title='Category Distribution by Subject',
                    labels={'value': 'Count', 'subject': 'Subject'})
        st.plotly_chart(fig)

        # Category prediction
        st.subheader("Predict Category")
        user_input = st.text_area("Enter feedback text to predict category:",
                                 "The assessment methods were well designed and fair.")

        if st.button("Predict"):
            # Preprocess input
            text_processor = TextProcessor()
            processed_input = text_processor.preprocess(user_input)

            # Transform to features
            input_features = vectorizer.transform([processed_input])

            # Predict
            prediction = category_model.predict(input_features)[0]

            # Display result
            st.success(f"Predicted category: {prediction}")

    # Topic Modeling page
    elif page == "Topic Modeling":
        st.title("Topic Modeling")

        # Preprocess data
        df = preprocess_data(df)

        # Parameters
        n_topics = st.slider("Number of topics", min_value=2, max_value=10, value=5)
        method = st.selectbox("Topic modeling method", ["LDA", "NMF"])

        # Perform topic modeling
        with st.spinner("Performing topic modeling..."):
            topic_modeler = TopicModeler(
                method=method.lower(),
                n_topics=n_topics
            )
            topic_modeler.fit(df['processed_text'])

            # Get top words
            top_words = topic_modeler.get_top_words_per_topic(n_words=10)

        # Display top words
        st.subheader(f"Top Words for Each Topic ({method})")

        for i, words in enumerate(top_words):
            st.write(f"**Topic {i+1}:** {', '.join(words)}")

        # Get topic distribution
        topic_dist = topic_modeler.get_topic_distribution(df['processed_text'])

        # Display topic distribution
        st.subheader("Topic Distribution")
        topic_counts = topic_dist['Dominant_Topic'].value_counts().reset_index()
        topic_counts.columns = ['Topic', 'Count']

        fig = px.bar(topic_counts, x='Topic', y='Count', title='Dominant Topic Distribution')
        st.plotly_chart(fig)

        # Display sample documents for each topic
        st.subheader("Sample Feedback by Topic")

        df_with_topics = df.copy()
        df_with_topics['Dominant_Topic'] = topic_dist['Dominant_Topic']

        for topic in topic_dist['Dominant_Topic'].unique():
            with st.expander(f"Topic {topic}"):
                samples = df_with_topics[df_with_topics['Dominant_Topic'] == topic]['feedback_text'].head(3)
                for i, sample in enumerate(samples):
                    st.write(f"{i+1}. {sample}")

    # Clustering page
    elif page == "Clustering":
        st.title("Clustering Analysis")

        # Preprocess data
        df = preprocess_data(df)

        # Extract features
        feature_matrix, vectorizer = extract_text_features(df)

        # Parameters
        n_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=5)
        method = st.selectbox("Clustering method", ["K-means", "Hierarchical"])

        # Perform clustering
        with st.spinner("Performing clustering..."):
            # Reduce dimensionality
            reducer = DimensionalityReducer(method='svd', n_components=50)
            reduced_features = reducer.fit_transform(feature_matrix)

            # Cluster
            clusterer = FeedbackClusterer(
                method=method.lower().replace('-', ''),
                n_clusters=n_clusters
            )
            clusterer.fit(reduced_features)

            # Get cluster labels
            cluster_labels = clusterer.model.labels_

        # Evaluate clustering
        results = clusterer.evaluate(reduced_features)

        # Display results
        st.subheader("Clustering Results")

        col1, col2 = st.columns(2)

        with col1:
            if results['silhouette_score'] is not None:
                st.metric("Silhouette Score", f"{results['silhouette_score']:.4f}")
            else:
                st.write("Silhouette score not available")

        with col2:
            st.metric("Number of Clusters", results['n_clusters'])

        # Display cluster sizes
        st.subheader("Cluster Sizes")
        cluster_sizes = pd.DataFrame({
            'Cluster': list(results['cluster_sizes'].keys()),
            'Size': list(results['cluster_sizes'].values())
        })

        fig = px.bar(cluster_sizes, x='Cluster', y='Size', title='Cluster Sizes')
        st.plotly_chart(fig)

        # Visualize clusters
        st.subheader("Cluster Visualization")

        # Reduce to 2D for visualization
        with st.spinner("Generating visualization..."):
            tsne_reducer = DimensionalityReducer(method='svd', n_components=2)
            features_2d = tsne_reducer.fit_transform(reduced_features)

            # Create DataFrame for plotting
            viz_df = pd.DataFrame({
                'x': features_2d[:, 0],
                'y': features_2d[:, 1],
                'cluster': cluster_labels
            })

            fig = px.scatter(viz_df, x='x', y='y', color='cluster',
                           title='Cluster Visualization',
                           labels={'cluster': 'Cluster'})
            st.plotly_chart(fig)

        # Display sample documents for each cluster
        st.subheader("Sample Feedback by Cluster")

        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = cluster_labels

        for cluster in range(n_clusters):
            with st.expander(f"Cluster {cluster}"):
                samples = df_with_clusters[df_with_clusters['cluster'] == cluster]['feedback_text'].head(3)
                for i, sample in enumerate(samples):
                    st.write(f"{i+1}. {sample}")

    # Combined Analysis page
    elif page == "Combined Analysis":
        st.title("Combined Analysis")

        # Preprocess data
        df = preprocess_data(df)

        # Extract features
        feature_matrix, vectorizer = extract_text_features(df)

        # Perform topic modeling
        topic_modeler = perform_topic_modeling(df['processed_text'], n_topics=5)
        topic_dist = topic_modeler.get_topic_distribution(df['processed_text'])

        # Perform clustering
        clusterer, reduced_features = perform_clustering(feature_matrix, n_clusters=5)
        cluster_labels = clusterer.model.labels_

        # Combine results
        combined_df = df.copy()
        combined_df['cluster'] = cluster_labels
        combined_df['dominant_topic'] = topic_dist['Dominant_Topic']

        # Analyze relationship between clusters and sentiment
        st.subheader("Sentiment Distribution by Cluster")

        cluster_sentiment = pd.crosstab(combined_df['cluster'], combined_df['true_sentiment'])
        cluster_sentiment_pct = cluster_sentiment.div(cluster_sentiment.sum(axis=1), axis=0)

        fig = px.bar(cluster_sentiment_pct.reset_index().melt(id_vars='cluster'),
                    x='cluster', y='value', color='true_sentiment',
                    title='Sentiment Distribution by Cluster',
                    labels={'value': 'Percentage', 'cluster': 'Cluster'})
        st.plotly_chart(fig)

        # Analyze relationship between topics and sentiment
        st.subheader("Sentiment Distribution by Topic")

        topic_sentiment = pd.crosstab(combined_df['dominant_topic'], combined_df['true_sentiment'])
        topic_sentiment_pct = topic_sentiment.div(topic_sentiment.sum(axis=1), axis=0)

        fig = px.bar(topic_sentiment_pct.reset_index().melt(id_vars='dominant_topic'),
                    x='dominant_topic', y='value', color='true_sentiment',
                    title='Sentiment Distribution by Topic',
                    labels={'value': 'Percentage', 'dominant_topic': 'Topic'})
        st.plotly_chart(fig)

        # Analyze relationship between topics and categories
        st.subheader("Category Distribution by Topic")

        topic_category = pd.crosstab(combined_df['dominant_topic'], combined_df['true_category'])
        topic_category_pct = topic_category.div(topic_category.sum(axis=1), axis=0)

        fig = px.bar(topic_category_pct.reset_index().melt(id_vars='dominant_topic'),
                    x='dominant_topic', y='value', color='true_category',
                    title='Category Distribution by Topic',
                    labels={'value': 'Percentage', 'dominant_topic': 'Topic'})
        st.plotly_chart(fig)

        # Analyze relationship between clusters and categories
        st.subheader("Category Distribution by Cluster")

        cluster_category = pd.crosstab(combined_df['cluster'], combined_df['true_category'])
        cluster_category_pct = cluster_category.div(cluster_category.sum(axis=1), axis=0)

        fig = px.bar(cluster_category_pct.reset_index().melt(id_vars='cluster'),
                    x='cluster', y='value', color='true_category',
                    title='Category Distribution by Cluster',
                    labels={'value': 'Percentage', 'cluster': 'Cluster'})
        st.plotly_chart(fig)

        # Insights
        st.subheader("Key Insights")
        st.write("""
        The combined analysis reveals relationships between:

        1. **Topics and Sentiment**: Some topics are more associated with positive/negative sentiment
        2. **Clusters and Categories**: Certain clusters contain feedback predominantly from specific categories
        3. **Topics and Categories**: The relationship between topics and categories helps validate our topic modeling

        These insights can help educational institutions:
        - Identify areas receiving negative feedback for targeted improvements
        - Understand common themes in student feedback
        - Track sentiment across different aspects of the educational experience
        """)

    # Time Series Analysis page
    elif page == "Time Series Analysis":
        st.title("Time Series Analysis")

        # Preprocess data
        df = preprocess_data(df)

        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])

        # Time period selection
        st.subheader("Select Time Period")

        # Get min and max dates
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", min_date)
        with col2:
            end_date = st.date_input("End Date", max_date)

        # Filter data by date range
        mask = (df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))
        filtered_df = df[mask]

        if len(filtered_df) == 0:
            st.warning("No data available for the selected date range.")
        else:
            st.success(f"Showing data from {start_date} to {end_date} ({len(filtered_df)} feedback entries)")

            # Time aggregation
            time_freq = st.selectbox(
                "Time Aggregation",
                ["Daily", "Weekly", "Monthly", "Quarterly"],
                index=2  # Default to Monthly
            )

            # Map selection to pandas frequency string
            freq_map = {
                "Daily": "D",
                "Weekly": "W",
                "Monthly": "M",
                "Quarterly": "Q"
            }
            freq = freq_map[time_freq]

            # Analysis type tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Feedback Volume", "Sentiment Trends", "Category Trends", "Rating Trends"])

            # Tab 1: Feedback Volume
            with tab1:
                st.subheader("Feedback Volume Over Time")

                # Group by date
                volume_data = filtered_df.groupby(pd.Grouper(key='date', freq=freq)).size().reset_index(name='count')

                # Plot feedback volume
                fig = px.line(volume_data, x='date', y='count', markers=True,
                             title=f'{time_freq} Feedback Volume')
                st.plotly_chart(fig)

                # Detect anomalies in feedback volume
                if len(volume_data) > 5:  # Need enough data points for anomaly detection
                    st.subheader("Anomaly Detection in Feedback Volume")

                    # Parameters for anomaly detection
                    window = st.slider("Window Size", min_value=2, max_value=10, value=3,
                                      help="Number of periods to use for rolling statistics")
                    threshold = st.slider("Threshold", min_value=1.0, max_value=5.0, value=2.0, step=0.5,
                                         help="Number of standard deviations to consider as anomaly")

                    # Detect anomalies
                    anomaly_data = detect_anomalies(volume_data, 'count', window=window, threshold=threshold)

                    # Plot with anomalies
                    fig = go.Figure()

                    # Add main line
                    fig.add_trace(go.Scatter(
                        x=anomaly_data['date'],
                        y=anomaly_data['count'],
                        mode='lines+markers',
                        name='Feedback Count',
                        line=dict(color='blue')
                    ))

                    # Add rolling mean
                    fig.add_trace(go.Scatter(
                        x=anomaly_data['date'],
                        y=anomaly_data['rolling_mean'],
                        mode='lines',
                        name='Rolling Mean',
                        line=dict(color='green', dash='dash')
                    ))

                    # Add bounds
                    fig.add_trace(go.Scatter(
                        x=anomaly_data['date'],
                        y=anomaly_data['upper_bound'],
                        mode='lines',
                        name='Upper Bound',
                        line=dict(color='rgba(0,100,0,0.2)'),
                        showlegend=False
                    ))

                    fig.add_trace(go.Scatter(
                        x=anomaly_data['date'],
                        y=anomaly_data['lower_bound'],
                        mode='lines',
                        name='Lower Bound',
                        line=dict(color='rgba(0,100,0,0.2)'),
                        fill='tonexty',
                        showlegend=False
                    ))

                    # Add positive anomalies
                    positive_anomalies = anomaly_data[anomaly_data['anomaly'] == 1]
                    if not positive_anomalies.empty:
                        fig.add_trace(go.Scatter(
                            x=positive_anomalies['date'],
                            y=positive_anomalies['count'],
                            mode='markers',
                            name='High Anomaly',
                            marker=dict(color='red', size=12, symbol='circle')
                        ))

                    # Add negative anomalies
                    negative_anomalies = anomaly_data[anomaly_data['anomaly'] == -1]
                    if not negative_anomalies.empty:
                        fig.add_trace(go.Scatter(
                            x=negative_anomalies['date'],
                            y=negative_anomalies['count'],
                            mode='markers',
                            name='Low Anomaly',
                            marker=dict(color='purple', size=12, symbol='circle')
                        ))

                    fig.update_layout(
                        title=f'Anomaly Detection in {time_freq} Feedback Volume',
                        xaxis_title='Date',
                        yaxis_title='Feedback Count'
                    )

                    st.plotly_chart(fig)

                    # Display anomalies in table
                    all_anomalies = anomaly_data[anomaly_data['anomaly'] != 0]
                    if not all_anomalies.empty:
                        st.subheader("Detected Anomalies")
                        anomaly_table = all_anomalies[['date', 'count', 'anomaly']].copy()
                        anomaly_table['anomaly_type'] = anomaly_table['anomaly'].apply(
                            lambda x: 'High Volume' if x == 1 else 'Low Volume')
                        st.dataframe(anomaly_table[['date', 'count', 'anomaly_type']])
                    else:
                        st.info("No anomalies detected with current settings.")
                else:
                    st.info("Not enough data points for anomaly detection. Try selecting a wider date range or a finer time aggregation.")

            # Tab 2: Sentiment Trends
            with tab2:
                st.subheader("Sentiment Trends Over Time")

                # Analyze sentiment over time
                sentiment_data = analyze_sentiment_over_time(filtered_df, freq=freq)

                # Plot sentiment counts
                st.write("Sentiment Counts")
                sentiment_columns = [col for col in sentiment_data.columns
                                   if col != 'date' and not col.endswith('_pct')]

                fig = go.Figure()
                for col in sentiment_columns:
                    fig.add_trace(go.Scatter(
                        x=sentiment_data['date'],
                        y=sentiment_data[col],
                        mode='lines+markers',
                        name=col
                    ))

                fig.update_layout(
                    title=f'{time_freq} Sentiment Counts',
                    xaxis_title='Date',
                    yaxis_title='Count'
                )

                st.plotly_chart(fig)

                # Plot sentiment percentages
                st.write("Sentiment Percentages")
                percentage_columns = [col for col in sentiment_data.columns
                                    if col.endswith('_pct')]

                fig = go.Figure()
                for col in percentage_columns:
                    sentiment_name = col.replace('_pct', '')
                    fig.add_trace(go.Scatter(
                        x=sentiment_data['date'],
                        y=sentiment_data[col],
                        mode='lines+markers',
                        name=sentiment_name
                    ))

                fig.update_layout(
                    title=f'{time_freq} Sentiment Percentages',
                    xaxis_title='Date',
                    yaxis_title='Percentage (%)',
                    yaxis=dict(range=[0, 100])
                )

                st.plotly_chart(fig)

                # Stacked area chart for sentiment percentages
                st.write("Sentiment Composition Over Time")

                fig = px.area(sentiment_data, x='date', y=percentage_columns,
                             title=f'{time_freq} Sentiment Composition',
                             labels={col: col.replace('_pct', '') for col in percentage_columns})

                st.plotly_chart(fig)

            # Tab 3: Category Trends
            with tab3:
                st.subheader("Category Trends Over Time")

                # Analyze categories over time
                category_data = analyze_categories_over_time(filtered_df, freq=freq)

                # Get category columns
                category_columns = [col for col in category_data.columns if col != 'date']

                # Plot category counts
                st.write("Category Counts")

                fig = go.Figure()
                for col in category_columns:
                    fig.add_trace(go.Scatter(
                        x=category_data['date'],
                        y=category_data[col],
                        mode='lines+markers',
                        name=col
                    ))

                fig.update_layout(
                    title=f'{time_freq} Category Counts',
                    xaxis_title='Date',
                    yaxis_title='Count'
                )

                st.plotly_chart(fig)

                # Stacked area chart for categories
                st.write("Category Composition Over Time")

                fig = px.area(category_data, x='date', y=category_columns,
                             title=f'{time_freq} Category Composition')

                st.plotly_chart(fig)

                # Heatmap of categories over time
                st.write("Category Heatmap Over Time")

                # Pivot data for heatmap
                pivot_data = category_data.set_index('date')

                fig = px.imshow(pivot_data.T,
                               labels=dict(x="Date", y="Category", color="Count"),
                               title=f'{time_freq} Category Heatmap',
                               color_continuous_scale='YlGnBu')

                st.plotly_chart(fig)

            # Tab 4: Rating Trends
            with tab4:
                st.subheader("Rating Trends Over Time")

                # Analyze ratings over time
                rating_data = analyze_ratings_over_time(filtered_df, freq=freq)

                # Plot average rating
                st.write("Average Rating Over Time")

                fig = go.Figure()

                # Add average rating line
                fig.add_trace(go.Scatter(
                    x=rating_data['date'],
                    y=rating_data['mean'],
                    mode='lines+markers',
                    name='Average Rating',
                    line=dict(color='blue')
                ))

                # Add standard deviation range if available
                if 'std' in rating_data.columns:
                    fig.add_trace(go.Scatter(
                        x=rating_data['date'],
                        y=rating_data['mean'] + rating_data['std'],
                        mode='lines',
                        name='Upper Std Dev',
                        line=dict(width=0),
                        showlegend=False
                    ))

                    fig.add_trace(go.Scatter(
                        x=rating_data['date'],
                        y=rating_data['mean'] - rating_data['std'],
                        mode='lines',
                        name='Lower Std Dev',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(0, 0, 255, 0.2)',
                        showlegend=False
                    ))

                fig.update_layout(
                    title=f'{time_freq} Average Rating',
                    xaxis_title='Date',
                    yaxis_title='Rating',
                    yaxis=dict(range=[1, 5])
                )

                st.plotly_chart(fig)

                # Plot feedback count
                st.write("Feedback Volume Over Time")

                fig = px.bar(rating_data, x='date', y='count',
                            title=f'{time_freq} Feedback Volume',
                            labels={'count': 'Number of Feedback', 'date': 'Date'})

                st.plotly_chart(fig)

                # Correlation between rating and volume
                if len(rating_data) > 5:
                    st.subheader("Correlation Analysis")

                    correlation = rating_data['mean'].corr(rating_data['count'])

                    st.write(f"Correlation between average rating and feedback volume: **{correlation:.4f}**")

                    if abs(correlation) > 0.5:
                        if correlation > 0:
                            st.info("There is a positive correlation between rating and volume, suggesting that students tend to provide more feedback when they are satisfied.")
                        else:
                            st.info("There is a negative correlation between rating and volume, suggesting that students tend to provide more feedback when they are dissatisfied.")
                    else:
                        st.info("There is no strong correlation between rating and feedback volume.")

    # Feedback Collection page
    elif page == "Feedback Collection":
        st.title("Feedback Collection")

        # Initialize feedback collector
        feedback_collector = FeedbackCollector()

        # Create tabs for submission and management
        tab1, tab2 = st.tabs(["Submit Feedback", "Manage Feedback"])

        # Tab 1: Submit Feedback
        with tab1:
            st.subheader("Submit New Feedback")

            # Create feedback form
            with st.form("feedback_form"):
                # Student information (optional)
                st.write("Student Information (Optional)")
                col1, col2 = st.columns(2)
                with col1:
                    student_id = st.text_input("Student ID", placeholder="Anonymous")
                with col2:
                    course_id = st.text_input("Course ID", placeholder="Unknown")

                # Subject selection
                subject = st.selectbox(
                    "Subject",
                    ["Mathematics", "Computer Science", "Physics", "Chemistry",
                     "Biology", "History", "Literature", "Economics", "Psychology", "Other"]
                )

                # Feedback text
                feedback_text = st.text_area(
                    "Feedback",
                    placeholder="Please provide your feedback here...",
                    height=150
                )

                # Rating
                rating = st.slider("Rating", min_value=1, max_value=5, value=3, step=1,
                                 help="1 = Very Dissatisfied, 5 = Very Satisfied")

                # Category selection (optional)
                category = st.selectbox(
                    "Category (Optional)",
                    [None, "Course Content", "Teaching Quality", "Assessment Methods",
                     "Learning Resources", "Facilities", "Support Services"]
                )

                # Submit button
                submit_button = st.form_submit_button("Submit Feedback")

            # Handle form submission
            if submit_button:
                if not feedback_text:
                    st.error("Please provide feedback text.")
                else:
                    # Prepare feedback data
                    feedback_data = {
                        'student_id': student_id if student_id else "anonymous",
                        'course_id': course_id if course_id else "unknown",
                        'subject': subject,
                        'feedback_text': feedback_text,
                        'rating': rating,
                        'category': category
                    }

                    # Try to predict sentiment if not provided
                    try:
                        # Preprocess text
                        text_processor = TextProcessor()
                        processed_text = text_processor.preprocess(feedback_text)

                        # Extract features
                        feature_matrix, vectorizer = extract_features([processed_text], method='tfidf')

                        # Train a simple sentiment model if not already available
                        X_train, X_test, y_train, y_test = train_test_split(
                            feature_matrix, df['true_sentiment'], test_size=0.3, random_state=42
                        )
                        sentiment_model = SentimentClassifier(model_type='logistic_regression')
                        sentiment_model.train(X_train, y_train)

                        # Predict sentiment
                        sentiment = sentiment_model.predict(feature_matrix)[0]
                        feedback_data['sentiment'] = sentiment
                    except Exception as e:
                        st.warning(f"Could not predict sentiment: {str(e)}")

                    # Submit feedback
                    result = feedback_collector.submit_feedback(feedback_data)

                    if result['status'] == 'success':
                        st.success(result['message'])
                        st.balloons()
                    else:
                        st.error(result['message'])

            # Display tips for providing effective feedback
            with st.expander("Tips for Providing Effective Feedback"):
                st.write("""
                1. **Be Specific**: Provide concrete examples rather than general statements.
                2. **Be Constructive**: Focus on how things can be improved.
                3. **Be Balanced**: Mention both positive aspects and areas for improvement.
                4. **Be Respectful**: Use appropriate language and tone.
                5. **Be Relevant**: Focus on aspects that can actually be changed or improved.
                """)

        # Tab 2: Manage Feedback
        with tab2:
            st.subheader("Manage Collected Feedback")

            # Get feedback statistics
            stats = feedback_collector.get_feedback_stats()

            # Display statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Feedback", stats['total_count'])
            with col2:
                if stats['average_rating'] is not None:
                    st.metric("Average Rating", f"{stats['average_rating']:.2f}")
                else:
                    st.metric("Average Rating", "N/A")
            with col3:
                sentiment_counts = stats['sentiment_counts']
                if sentiment_counts and 'Positive' in sentiment_counts:
                    positive_pct = sentiment_counts.get('Positive', 0) / stats['total_count'] * 100 if stats['total_count'] > 0 else 0
                    st.metric("Positive Feedback", f"{positive_pct:.1f}%")

            # Get collected feedback
            collected_feedback = feedback_collector.get_feedback()

            if len(collected_feedback) > 0:
                # Display feedback table
                st.subheader("Collected Feedback")
                st.dataframe(collected_feedback)

                # Allow downloading the feedback as CSV
                csv = collected_feedback.to_csv(index=False)
                st.download_button(
                    label="Download Feedback as CSV",
                    data=csv,
                    file_name="collected_feedback.csv",
                    mime="text/csv"
                )

                # Visualize feedback
                st.subheader("Feedback Visualizations")

                col1, col2 = st.columns(2)

                with col1:
                    # Rating distribution
                    if 'rating' in collected_feedback.columns:
                        fig = px.histogram(collected_feedback, x='rating',
                                          title='Rating Distribution',
                                          labels={'rating': 'Rating', 'count': 'Count'},
                                          nbins=5)
                        st.plotly_chart(fig)

                with col2:
                    # Sentiment distribution
                    if 'sentiment' in collected_feedback.columns and not collected_feedback['sentiment'].isna().all():
                        fig = px.pie(collected_feedback, names='sentiment',
                                    title='Sentiment Distribution')
                        st.plotly_chart(fig)

                # Subject distribution
                if 'subject' in collected_feedback.columns:
                    # Get subject counts and prepare for plotting
                    subject_counts = collected_feedback['subject'].value_counts().reset_index()
                    subject_counts.columns = ['Subject', 'Count']

                    # Create the bar chart with correct column names
                    fig = px.bar(subject_counts,
                                x='Subject', y='Count', title='Subject Distribution')
                    st.plotly_chart(fig)

                # Category distribution
                if 'category' in collected_feedback.columns and not collected_feedback['category'].isna().all():
                    # Get category counts and prepare for plotting
                    category_counts = collected_feedback['category'].value_counts().reset_index()
                    category_counts.columns = ['Category', 'Count']

                    # Create the bar chart with correct column names
                    fig = px.bar(category_counts,
                                x='Category', y='Count', title='Category Distribution')
                    st.plotly_chart(fig)

                # Word cloud of feedback text
                if 'feedback_text' in collected_feedback.columns:
                    st.subheader("Word Cloud of Collected Feedback")

                    # Create word cloud
                    wc_text = " ".join(collected_feedback['feedback_text'].dropna())
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wc_text)

                    # Display word cloud
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
            else:
                st.info("No feedback collected yet. Use the 'Submit Feedback' tab to add new feedback.")

    # User Management page (admin only)
    elif page == "User Management" and get_user_role() == 'admin':
        st.title("User Management")

        # Create tabs for different user management functions
        tab1, tab2 = st.tabs(["User List", "Create User"])

        # Tab 1: User List
        with tab1:
            st.subheader("Registered Users")

            # Get list of users
            users = auth.list_users()

            if users:
                # Convert to DataFrame for display
                users_df = pd.DataFrame(users)

                # Format dates
                if 'created_at' in users_df.columns:
                    users_df['created_at'] = pd.to_datetime(users_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
                if 'last_login' in users_df.columns:
                    users_df['last_login'] = pd.to_datetime(users_df['last_login'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')

                # Display users
                st.dataframe(users_df)

                # User details and actions
                st.subheader("User Details and Actions")

                # Select user
                selected_username = st.selectbox("Select User", [u['username'] for u in users])

                # Get selected user details
                selected_user = next((u for u in users if u['username'] == selected_username), None)

                if selected_user:
                    # Display user details
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**Username:** {selected_user['username']}")
                        st.write(f"**Full Name:** {selected_user['full_name'] or 'Not provided'}")
                        st.write(f"**Email:** {selected_user['email']}")

                    with col2:
                        st.write(f"**Role:** {selected_user['role']}")
                        st.write(f"**Created:** {selected_user['created_at']}")
                        st.write(f"**Last Login:** {selected_user['last_login'] or 'Never'}")

                    # User actions
                    st.subheader("Actions")

                    # Cannot modify own account here
                    if selected_user['username'] == st.session_state.user['username']:
                        st.warning("You cannot modify your own account from this page. Use the profile settings instead.")
                    else:
                        # Change role
                        new_role = st.selectbox("Change Role", ["admin", "instructor", "student"],
                                              index=["admin", "instructor", "student"].index(selected_user['role']))

                        if st.button("Update Role"):
                            result = auth.update_user(selected_user['username'], {'role': new_role})
                            if result['status'] == 'success':
                                st.success(result['message'])
                                st.rerun()
                            else:
                                st.error(result['message'])

                        # Delete user
                        st.warning("Danger Zone")
                        if st.button("Delete User", key="delete_user"):
                            st.session_state.confirm_delete = True

                        if st.session_state.get('confirm_delete', False):
                            st.error(f"Are you sure you want to delete user {selected_user['username']}? This action cannot be undone.")
                            col1, col2 = st.columns(2)

                            with col1:
                                if st.button("Yes, Delete"):
                                    # For safety, require admin password
                                    admin_password = st.text_input("Enter your admin password to confirm", type="password")
                                    if admin_password:
                                        result = auth.delete_user(selected_user['username'], admin_password)
                                        if result['status'] == 'success':
                                            st.success(result['message'])
                                            st.session_state.confirm_delete = False
                                            st.rerun()
                                        else:
                                            st.error(result['message'])

                            with col2:
                                if st.button("Cancel"):
                                    st.session_state.confirm_delete = False
                                    st.rerun()
            else:
                st.info("No users found.")

        # Tab 2: Create User
        with tab2:
            st.subheader("Create New User")

            with st.form("create_user_form"):
                new_username = st.text_input("Username")
                new_password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                email = st.text_input("Email")
                full_name = st.text_input("Full Name")
                role = st.selectbox("Role", ["admin", "instructor", "student"])

                create_button = st.form_submit_button("Create User")

                if create_button:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif not new_username or not new_password or not email:
                        st.error("Please fill in all required fields")
                    else:
                        result = auth.register_user(
                            username=new_username,
                            password=new_password,
                            email=email,
                            role=role,
                            full_name=full_name
                        )

                        if result['status'] == 'success':
                            st.success(result['message'])
                        else:
                            st.error(result['message'])

if __name__ == "__main__":
    main()
