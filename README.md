# 🎯 Amazon Product Review Sentiment Analyzer

A machine learning web application that analyzes sentiment of Amazon product reviews using Natural Language Processing and Logistic Regression classification.

## 🌟 Live Demo

🚀 **Try it live:** [Coming Soon - Deploy to Heroku/Railway]
|
## 🎯 Features

- **🤖 Machine Learning** - Logistic Regression model with 82.3% accuracy
- **📝 Text Preprocessing** - Advanced NLP with NLTK tokenization & lemmatization
- **🎭 Sentiment Analysis** - Classifies reviews as Positive/Negative with confidence scores
- **🌐 Web Interface** - Beautiful Flask web app with Bootstrap UI
- **🔗 REST API** - JSON API for integration with other applications
- **📊 Real-time Analysis** - Instant predictions with detailed confidence metrics

## 🛠️ Technology Stack

**Backend:**
- Python 3.9
- Flask 2.3
- Scikit-learn 1.3
- NLTK 3.8
- Pandas & NumPy

**Frontend:**
- HTML5, CSS3
- Bootstrap 5
- Jinja2 Templates
- Font Awesome Icons

**Machine Learning:**
- Logistic Regression Classifier
- TF-IDF Vectorization
- NLTK for text preprocessing
- Cross-validation evaluation



1. # Create conda environment
conda create -n sentiment_flask python=3.9

2. # Activate environment
conda activate sentiment_flask

3. # Install essential packages
pip install -r requirements.txt

4.Download Dataset (See Dataset section below)

5.Train the model
python training/dataset_preparation.py
python training/train_model.py

6.Run the application
python run.py

7.Open your browser
http://localhost:5000



### 📊 Dataset Details
This project uses the Amazon Fine Food Reviews dataset from Kaggle.

- **Source**: [Kaggle - Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

📞 Contact
K ADITHYA OM - @kadithyaom@gmail.com

Project Link: https://github.com/adithyaom18/amazon-sentiment-analyzer