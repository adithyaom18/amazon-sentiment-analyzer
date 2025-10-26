import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import sys
import os

# Add the app directory to Python path to import preprocessing
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from app.models.preprocessing import TextPreprocessor

class ModelTrainer:
    def __init__(self):
        self.preprocessor = TextPreprocessor()  # Now using the imported class
        self.vectorizer = None
        self.model = None
        self.feature_method = None
        
    def load_data(self, use_sample=True):
        """
        Load training data
        """
        try:
            if use_sample:
                df = pd.read_csv('data/sample_reviews.csv')
                print(f"📊 Using sample dataset: {len(df)} reviews")
            else:
                df = pd.read_csv('data/processed_reviews.csv')
                print(f"📊 Using full dataset: {len(df)} reviews")
            
            print(f"🎭 Sentiment distribution:\n{df['sentiment'].value_counts()}")
            return df
            
        except FileNotFoundError:
            print("❌ No dataset found. Run dataset_preparation.py first.")
            return None
    
    def extract_features(self, texts, method='tfidf'):
        """
        Feature extraction using different methods
        """
        print(f"🔧 Extracting features using {method.upper()}...")
        
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),  # Using unigrams and bigrams
                stop_words='english',
                min_df=2,
                max_df=0.85
            )
            features = self.vectorizer.fit_transform(texts)
            
        elif method == 'bow':
            self.vectorizer = CountVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english'
            )
            features = self.vectorizer.fit_transform(texts)
            
        elif method == 'tfidf_advanced':
            self.vectorizer = TfidfVectorizer(
                max_features=8000,
                ngram_range=(1, 3),  # uni, bi, and tri-grams
                min_df=2,
                max_df=0.85,
                stop_words='english',
                sublinear_tf=True
            )
            features = self.vectorizer.fit_transform(texts)
        
        self.feature_method = method
        print(f"✅ Feature shape: {features.shape}")
        return features
    
    def preprocess_data(self, df):
        """
        Preprocess the text data
        """
        print("🔄 Preprocessing text data...")
        
        # Apply preprocessing pipeline
        df['processed_text'] = df['review'].apply(self.preprocessor.preprocess_pipeline)
        
        print("✅ Text preprocessing completed")
        print(f"📝 Sample processed text: {df['processed_text'].iloc[0][:100]}...")
        
        return df
    
    def train_and_compare_models(self, X, y):
        """
        Train and compare multiple ML models
        """
        print("\n🤖 Training and comparing models...")
        print("=" * 50)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define models to compare
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='linear', probability=True, random_state=42),
            'Naive Bayes': MultinomialNB()
        }
        
        best_model = None
        best_score = 0
        best_model_name = ""
        results = {}
        
        for name, model in models.items():
            print(f"\n🔍 Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred
                }
                
                print(f"✅ {name} - Accuracy: {accuracy:.4f}")
                print(f"📊 Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
                # Update best model
                if accuracy > best_score:
                    best_score = accuracy
                    best_model = model
                    best_model_name = name
                    
            except Exception as e:
                print(f"❌ Error training {name}: {e}")
                continue
        
        self.model = best_model
        print(f"\n🎯 Best Model: {best_model_name} with accuracy: {best_score:.4f}")
        
        return results, X_test, y_test, best_model_name
    
    def evaluate_models(self, results, X_test, y_test, best_model_name):
        """
        Comprehensive model evaluation
        """
        print("\n📊 Model Evaluation Results")
        print("=" * 60)
        
        # Create comparison table
        comparison_data = []
        for name, result in results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'CV Score': f"{result['cv_mean']:.4f} (±{result['cv_std'] * 2:.4f})"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        print("\n🏆 Model Performance Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Detailed report for best model
        best_result = results[best_model_name]
        print(f"\n📈 Detailed Report for {best_model_name}:")
        print(classification_report(y_test, best_result['predictions']))
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, best_result['predictions'], best_model_name)
        
        return comparison_df
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """
        Plot confusion matrix
        """
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Save the plot
        os.makedirs('static/images', exist_ok=True)
        plt.savefig('static/images/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📈 Confusion matrix saved as 'static/images/confusion_matrix.png'")
    
    def save_model(self):
        """
        Save the trained model and vectorizer
        """
        if self.model and self.vectorizer:
            os.makedirs('models', exist_ok=True)
            
            model_path = f'models/sentiment_model_{self.feature_method}.pkl'
            vectorizer_path = f'models/vectorizer_{self.feature_method}.pkl'
            
            joblib.dump(self.model, model_path)
            joblib.dump(self.vectorizer, vectorizer_path)
            
            print(f"\n💾 Model saved: {model_path}")
            print(f"💾 Vectorizer saved: {vectorizer_path}")
        else:
            print("❌ No model or vectorizer to save")
    
    def run_training_pipeline(self, use_sample=True, feature_method='tfidf'):
        """
        Run complete training pipeline
        """
        print("🚀 Starting Model Training Pipeline")
        print("=" * 60)
        
        # 1. Load data
        df = self.load_data(use_sample=use_sample)
        if df is None:
            return
        
        # 2. Preprocess data
        df = self.preprocess_data(df)
        
        # 3. Extract features
        X = self.extract_features(df['processed_text'], method=feature_method)
        y = df['sentiment']
        
        # 4. Train and compare models
        results, X_test, y_test, best_model_name = self.train_and_compare_models(X, y)
        
        # 5. Evaluate models
        comparison_df = self.evaluate_models(results, X_test, y_test, best_model_name)
        
        # 6. Save the best model
        self.save_model()
        
        print("\n✅ Training pipeline completed successfully!")
        return comparison_df

if __name__ == "__main__":
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Change this line from True to False
    print("🎯 Training with FULL dataset (better accuracy)...")
    results = trainer.run_training_pipeline(use_sample=False, feature_method='tfidf')