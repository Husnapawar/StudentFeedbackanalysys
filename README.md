# Student Feedback Analysis System

## Overview
This project implements a comprehensive system for analyzing student feedback using both supervised and unsupervised machine learning techniques. The system processes textual feedback data, extracts insights, categorizes feedback, and evaluates performance metrics to provide a holistic understanding of student opinions and experiences.

## Features
- Text preprocessing and feature extraction
- Sentiment analysis of feedback (positive/negative/neutral)
- Feedback categorization into predefined topics
- Topic modeling to discover hidden themes
- Clustering similar feedback
- Performance evaluation metrics
- Interactive visualizations and reports
- Sample data generation for testing and development

## Project Structure
```
SFD/
├── data/                  # For storing raw and processed feedback data
├── notebooks/             # Jupyter notebooks for exploration and visualization
│   ├── student_feedback_analysis.ipynb    # Main analysis notebook
│   └── unsupervised_learning.ipynb        # Unsupervised learning techniques
├── src/                   # Source code
│   ├── preprocessing/     # Data cleaning and preprocessing
│   │   └── text_processor.py              # Text processing utilities
│   ├── models/            # ML models implementation
│   │   ├── supervised_models.py           # Sentiment and category classifiers
│   │   └── unsupervised_models.py         # Topic modeling and clustering
│   ├── evaluation/        # Performance metrics and evaluation
│   │   └── metrics.py                     # Evaluation metrics
│   ├── visualization/     # Visualization utilities
│   │   └── visualizer.py                  # Visualization functions
│   └── utils/             # Helper functions
│       └── data_generator.py              # Sample data generation
├── tests/                 # Unit tests
│   └── test_models.py                     # Tests for ML models
├── results/               # Output directory for visualizations and results
├── requirements.txt       # Project dependencies
├── run_tests.py           # Script to run all tests
├── src/main.py            # Main script to run the complete analysis
└── README.md              # Project documentation
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup
1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd SFD
   ```

2. Create a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Generate Sample Data
If you don't have your own feedback data, you can generate synthetic data for testing:

```bash
# Run the data generator script
python -c "from src.utils.data_generator import generate_sample_feedback; generate_sample_feedback(n_samples=1000, output_path='data/sample_feedback.csv')"
```

### Run the Complete Analysis
To run the complete analysis pipeline:

```bash
python src/main.py
```

This will:
1. Load or generate sample data
2. Preprocess the text data
3. Train supervised models for sentiment analysis and categorization
4. Perform unsupervised learning (topic modeling and clustering)
5. Evaluate model performance
6. Generate visualizations in the `results/` directory

### Interactive Dashboard
To launch the interactive web dashboard:

```bash
streamlit run dashboard.py
```

The dashboard provides:
- Interactive data exploration
- Sentiment analysis visualization
- Feedback categorization insights
- Topic modeling results
- Clustering visualization
- Time series analysis and trend tracking
- Anomaly detection in feedback patterns
- Real-time feedback collection and management
- User authentication and role-based access control
- User management for administrators
- Combined analysis of supervised and unsupervised results

### Interactive Exploration with Jupyter Notebooks
For interactive exploration and visualization:

```bash
jupyter notebook notebooks/
```

Two notebooks are provided:
- `student_feedback_analysis.ipynb`: General analysis and supervised learning
- `unsupervised_learning.ipynb`: Focused on topic modeling and clustering

### Run Tests
To verify that all components are working correctly:

```bash
python run_tests.py
```

## Customizing for Your Data

### Using Your Own Data
1. Prepare your feedback data in CSV format with at least a column named `feedback_text`
2. Optional columns for supervised learning: `sentiment`, `category`
3. Use the custom analysis script:

```bash
python custom_analysis.py path/to/your/data.csv --output results/your_analysis
```

This script will:
- Preprocess your feedback text
- Perform topic modeling to discover themes
- Cluster similar feedback
- Train and evaluate sentiment/category classifiers if labels are provided
- Save results and visualizations to the specified output directory

### Adjusting Model Parameters
You can customize various parameters in the code:

- **Text Preprocessing**: Modify stopwords, stemming/lemmatization options in `TextProcessor`
- **Feature Extraction**: Adjust n-gram range, max features in `extract_features`
- **Supervised Models**: Change model types, hyperparameters in `SentimentClassifier` and `FeedbackCategorizer`
- **Topic Modeling**: Adjust number of topics, method (LDA/NMF) in `TopicModeler`
- **Clustering**: Modify number of clusters, method in `FeedbackClusterer`

## Supervised Learning Components

### Sentiment Analysis
Classifies feedback as positive, neutral, or negative using various algorithms:
- Logistic Regression
- Naive Bayes
- Random Forest
- Support Vector Machines

### Feedback Categorization
Multi-class classification to categorize feedback into predefined topics such as:
- Course Content
- Teaching Quality
- Assessment Methods
- Learning Resources
- Facilities
- Support Services

## Unsupervised Learning Components

### Topic Modeling
Discovers hidden themes in the feedback using:
- Latent Dirichlet Allocation (LDA)
- Non-negative Matrix Factorization (NMF)

### Clustering
Groups similar feedback together using:
- K-means clustering
- Hierarchical clustering
- DBSCAN (for anomaly detection)

### Dimensionality Reduction
Reduces feature space for visualization and improved clustering:
- Truncated SVD
- NMF
- t-SNE (for visualization)

## Evaluation Metrics

### Classification Metrics
- Accuracy: Overall correctness
- Precision: Exactness of positive predictions
- Recall: Completeness of positive predictions
- F1-score: Harmonic mean of precision and recall
- Confusion Matrix: Detailed breakdown of predictions

### Clustering Evaluation
- Silhouette Score: Measure of cluster separation
- Davies-Bouldin Index: Ratio of within-cluster to between-cluster distances
- Calinski-Harabasz Index: Ratio of between-cluster to within-cluster dispersion

### Topic Modeling Evaluation
- Topic Coherence: Semantic similarity of words within topics
- Perplexity: How well the model predicts a sample

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
