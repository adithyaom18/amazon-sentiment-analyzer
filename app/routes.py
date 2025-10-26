from flask import render_template, request, jsonify
import joblib
import pandas as pd
import os
from app.models.preprocessing import TextPreprocessor
from app import app

# Initialize model and preprocessor
model = None
vectorizer = None
preprocessor = TextPreprocessor()

def load_model():
    """Load the trained model and vectorizer"""
    global model, vectorizer
    try:
        model_path = 'models/sentiment_model_tfidf.pkl'
        vectorizer_path = 'models/vectorizer_tfidf.pkl'
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            print("✅ Model and vectorizer loaded successfully")
        else:
            print("❌ Model files not found. Please train the model first.")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")

def get_model_info():
    """Get current model information including accuracy"""
    try:
        # Update this with your actual accuracy from training
        return {
            'name': 'Logistic Regression',
            'accuracy': '82.3%',  # Change this to your actual new accuracy
            'training_samples': '10,000 reviews',  # Updated to 10K
            'features': 'TF-IDF with unigrams and bigrams'
        }
    except:
        # Fallback info
        return {
            'name': 'Logistic Regression',
            'accuracy': '82.3%',
            'training_samples': '10,000 reviews',
            'features': 'TF-IDF with unigrams and bigrams'
        }

# Load model when the app starts
load_model()

@app.route('/')
def home():
    """Home page with sentiment analysis form"""
    model_info = get_model_info()
    return render_template('index.html', model_info=model_info)

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    """Analyze sentiment of submitted text"""
    try:
        text = request.form.get('text', '').strip()
        
        if not text:
            model_info = get_model_info()
            return render_template('index.html', 
                                 error="Please enter some text to analyze",
                                 model_info=model_info)
        
        if len(text) < 10:
            model_info = get_model_info()
            return render_template('index.html', 
                                 error="Please enter at least 10 characters",
                                 model_info=model_info)
        
        # Check if model is loaded
        if model is None or vectorizer is None:
            model_info = get_model_info()
            return render_template('index.html', 
                                 error="Model not loaded. Please train the model first.",
                                 model_info=model_info)
        
        # Preprocess the text
        processed_text = preprocessor.preprocess_pipeline(text)
        
        # Vectorize the text
        text_vector = vectorizer.transform([processed_text])
        
        # Predict sentiment
        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector)[0]
        
        # Get confidence scores
        positive_confidence = probability[1]
        negative_confidence = probability[0]
        
        # Determine sentiment and confidence
        sentiment = "positive" if prediction == 1 else "negative"
        confidence = positive_confidence if prediction == 1 else negative_confidence
        
        # Format results
        result = {
            'text': text,
            'sentiment': sentiment,
            'confidence': round(confidence * 100, 2),
            'positive_score': round(positive_confidence * 100, 2),
            'negative_score': round(negative_confidence * 100, 2),
            'processed_text': processed_text
        }
        
        # Pass model_info to results template
        model_info = get_model_info()
        return render_template('results.html', result=result, model_info=model_info)
    
    except Exception as e:
        model_info = get_model_info()
        return render_template('index.html', error=f"Error analyzing sentiment: {str(e)}", model_info=model_info)

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for sentiment analysis"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({'error': 'Text is empty'}), 400
        
        if len(text) < 10:
            return jsonify({'error': 'Text too short (min 10 characters)'}), 400
        
        # Check if model is loaded
        if model is None or vectorizer is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Preprocess the text
        processed_text = preprocessor.preprocess_pipeline(text)
        
        # Vectorize the text
        text_vector = vectorizer.transform([processed_text])
        
        # Predict sentiment
        prediction = model.predict(text_vector)[0]
        probability = model.predict_proba(text_vector)[0]
        
        # Get confidence scores
        positive_confidence = probability[1]
        negative_confidence = probability[0]
        
        # Determine sentiment and confidence
        sentiment = "positive" if prediction == 1 else "negative"
        confidence = positive_confidence if prediction == 1 else negative_confidence
        
        result = {
            'text': text,
            'sentiment': sentiment,
            'confidence': round(confidence * 100, 2),
            'positive_score': round(positive_confidence * 100, 2),
            'negative_score': round(negative_confidence * 100, 2),
            'processed_text': processed_text,
            'success': True
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/about')
def about():
    """About page"""
    model_info = get_model_info()
    return render_template('about.html', model_info=model_info)