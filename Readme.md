# 🧠 Hyperlocal News Anomaly Detection & Source Attribution Dashboard

## 🚀 Overview
This project detects **anomalous or misleading news articles** from hyperlocal data sources using advanced **Natural Language Processing (NLP)** and **Machine Learning** techniques.  
It integrates **sentiment analysis**, **location extraction**, **BERT embeddings**, and **anomaly detection models** to flag potential misinformation and visualize insights interactively through a Streamlit dashboard.

---

## 📁 Project Structure

Data_Science_project_6_Hyperlocal-News-Anomaly-Detection-and-Source-Attribution/
├── data/
│   └── Articles.csv                # Raw dataset
│
├── notebooks/
│   ├── models/                     # Trained ML models (Isolation Forest, BERT)
│   └── outputs/                    # Processed outputs and model results
│
├── streamlit/
│   └── streamlit_app.py            # Streamlit dashboard script
│
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation (this file)

---

## ⚙️ Features

| Component                  | Description                                                         |
| -------------------------- | ------------------------------------------------------------------- |
| 🧹 **Data Cleaning**       | Tokenization, stopword removal, lemmatization using spaCy           |
| 📍 **Location Extraction** | Extracts location entities using GeoText and spaCy NER              |
| 💬 **Sentiment Analysis**  | Calculates polarity scores using VaderSentiment                     |
| 🔍 **Anomaly Detection**   | Uses Isolation Forest to detect suspicious or inconsistent articles |
| 🧠 **Topic Modeling**      | Groups articles with BERTopic for contextual understanding          |
| 🌍 **Source Attribution**  | Uses Sentence Transformers (BERT) for classification                |
| 📊 **Visualization**       | Interactive plots and maps using Streamlit + Plotly                 |

---

## 🧰 Tech Stack

Language: Python 3.13.5

Core Libraries:
pandas, numpy, scikit-learn, plotly, streamlit, spacy,
vaderSentiment, bertopic, sentence-transformers, geotext, tqdm, joblib

Framework: Streamlit

Models Used: Logistic Regression, Isolation Forest, BERT-based Embeddings

---

## 🏗️ Setup Instructions

### 1️⃣ Clone the Repository

git clone https://github.com/Abu-git27/Data_Science_project_6_Hyperlocal-News-Anomaly-Detection-and-Source-Attribution.git
cd Data_Science_project_6_Hyperlocal-News-Anomaly-Detection-and-Source-Attribution

### 2️⃣ Install Dependencies

If you have a requirements.txt file:

pip install -r requirements.txt

Or install manually:

pip install pandas numpy scikit-learn spacy plotly streamlit vaderSentiment bertopic sentence-transformers geotext tqdm joblib chardet
python -m spacy download en_core_web_sm


### ▶️ Run the Dashboard

Launch the Streamlit application with:

streamlit run streamlit_app.py

---

## 📂 Key Files

| File                                            | Description                                                |
| ----------------------------------------------- | ---------------------------------------------------------- |
| `data/Articles.csv`                             | Dataset containing hyperlocal news articles                |
| `notebooks/outputs/processed_news.csv`          | Cleaned data with derived sentiment, topics, and anomalies |
| `notebooks/models/isolation_forest_model.pkl`   | Trained Isolation Forest model                             |
| `notebooks/models/location_classifier_bert.pkl` | BERT model for source classification                       |
| `streamlit_app.py`                              | Interactive Streamlit dashboard                            |
| `requirements.txt`                              | Dependencies list                                          |
| `README.md`                                     | Documentation file                                         |


---

## 📊 Dashboard Visuals

The Streamlit dashboard provides:

🧾 Article Summaries — Overview of processed articles

💬 Sentiment Distribution — Visualization of article sentiment

🌍 Regional Mapping — Geo-locations of news entities

🔎 Anomalous Article Detection — Outliers flagged visually

🧩 Topic Clustering — Topics discovered via BERTopic

---

## 🧠 Workflow

1. Load dataset
2. Clean and preprocess text
3. Extract entities and locations
4. Compute sentiment & embeddings
5. Apply anomaly detection
6. Visualize insights in Streamlit

---

## 👨‍💻 Author

Abu Shakeer
🎓 Capstone Project — GUVI Data Science Program
📧 abushakeer2002@gmail.com
🌐 GitHub: Abu-git27

---

## 🪄 Acknowledgements

spaCy
 — NLP processing

Streamlit
 — App framework

BERTopic
 — Topic modeling

VADER Sentiment
 — Sentiment analysis

 ---

 ## 🧾 License

This project is developed for educational and research purposes only