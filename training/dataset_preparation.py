import pandas as pd
import numpy as np
import nltk
from sklearn.utils import shuffle
import os

def download_nltk_resources():
    """Download required NLTK data"""
    print("📥 Downloading NLTK resources...")
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        print("✅ NLTK resources downloaded successfully")
    except Exception as e:
        print(f"❌ Error downloading NLTK resources: {e}")

def create_sample_dataset():
    """
    Create a sample dataset for development and testing
    This is GitHub-friendly and works without the large Amazon dataset
    """
    print("📝 Creating sample dataset for development...")
    
    # Realistic Amazon-style product reviews
    sample_reviews = [
        # Positive reviews (Score 4-5)
        "This product is absolutely fantastic! The quality exceeded my expectations and it works perfectly.",
        "Excellent value for money. Fast shipping and great customer service. Will definitely buy again.",
        "Love this product! It has all the features I needed and the build quality is outstanding.",
        "Best purchase I've made this year. The performance is incredible and it looks great too.",
        "Very impressed with this product. It arrived quickly and works exactly as described.",
        "Outstanding quality and great features. Much better than I expected for the price.",
        "Perfect product for my needs. Easy to use and very reliable. Five stars!",
        "Excellent product with fantastic performance. Would recommend to anyone.",
        "Great value and excellent quality. Very happy with this purchase.",
        "This is exactly what I was looking for. Works perfectly and great quality.",
        
        # Negative reviews (Score 1-2)  
        "Terrible product. Stopped working after just 2 days. Complete waste of money.",
        "Poor quality and disappointing performance. Would not recommend to anyone.",
        "Very disappointed with this purchase. The product does not work as advertised.",
        "Low quality materials and bad design. Avoid this product at all costs.",
        "Waste of money. The product broke immediately and customer service was unhelpful.",
        "Terrible quality. Much worse than expected. Would not buy again.",
        "Poor craftsmanship and cheap materials. Very disappointed.",
        "Does not work properly. Save your money and look elsewhere.",
        "Horrible product quality. Fell apart after minimal use.",
        "Complete garbage. Do not waste your time or money on this product."
    ]
    
    # Create balanced dataset (10 positive, 10 negative)
    sample_data = {
        'review': sample_reviews,
        'sentiment': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 1 = Positive
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # 0 = Negative
    }
    
    df = pd.DataFrame(sample_data)
    print(f"✅ Sample dataset created with {len(df)} reviews")
    print(f"📊 Sentiment distribution:\n{df['sentiment'].value_counts()}")
    
    return df

def load_amazon_data_if_available():
    """
    Try to load the actual Amazon dataset if available
    If not, use the sample dataset
    """
    try:
        # Try to load the Amazon dataset
        df = pd.read_csv('data/Reviews.csv')
        print(f"✅ Amazon dataset loaded: {len(df)} reviews")
        
        # Create sentiment labels (4-5 stars = Positive, 1-2 stars = Negative)
        df['sentiment'] = df['Score'].apply(lambda x: 1 if x > 3 else (0 if x < 3 else -1))
        
        # Remove neutral reviews (Score = 3)
        df = df[df['sentiment'] != -1]
        
        # Balance the dataset
        positive = df[df['sentiment'] == 1]
        negative = df[df['sentiment'] == 0]
        
        # Take smaller sample for development
        sample_size = min(5000, len(positive), len(negative))
        positive_sample = positive.sample(sample_size, random_state=42)
        negative_sample = negative.sample(sample_size, random_state=42)
        
        df_balanced = pd.concat([positive_sample, negative_sample])
        df_balanced = shuffle(df_balanced, random_state=42).reset_index(drop=True)
        
        # Rename 'Text' column to 'review' for consistency
        df_balanced = df_balanced.rename(columns={'Text': 'review'})
        
        print(f"📊 Balanced dataset: {len(df_balanced)} reviews ({sample_size} each class)")
        return df_balanced
        
    except FileNotFoundError:
        print("📝 Amazon dataset not found. Using sample dataset for development.")
        return create_sample_dataset()

def analyze_dataset(df):
    """
    Analyze and display dataset statistics
    """
    print("\n📈 Dataset Analysis:")
    print("=" * 50)
    print(f"Total reviews: {len(df)}")
    print(f"Positive reviews: {len(df[df['sentiment'] == 1])}")
    print(f"Negative reviews: {len(df[df['sentiment'] == 0])}")
    
    # Calculate average review length
    df['review_length'] = df['review'].apply(len)
    
    avg_length = df['review_length'].mean()
    print(f"Average review length: {avg_length:.1f} characters")
    
    # Show sample reviews
    print("\n🔍 Sample reviews from dataset:")
    print("-" * 30)
    for i, (_, row) in enumerate(df.head(3).iterrows(), 1):
        sentiment = "Positive" if row['sentiment'] == 1 else "Negative"
        review_text = row['review']
        print(f"Review {i} ({sentiment}): {review_text[:100]}...")

def save_datasets(df):
    """
    Save processed datasets for training
    """
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save full processed dataset
    df.to_csv('data/processed_reviews.csv', index=False)
    
    # Save a smaller version for quick testing
    df_small = df.sample(min(1000, len(df)), random_state=42)
    df_small.to_csv('data/sample_reviews.csv', index=False)
    
    print("💾 Datasets saved:")
    print(f"   - data/processed_reviews.csv ({len(df)} reviews)")
    print(f"   - data/sample_reviews.csv ({len(df_small)} reviews)")
    
    # Show column names for debugging
    print(f"📋 Columns in saved dataset: {df.columns.tolist()}")

if __name__ == "__main__":
    print("🚀 Starting Data Preparation Pipeline")
    print("=" * 60)
    
    # 1. Download NLTK resources
    download_nltk_resources()
    
    # 2. Load or create dataset
    df = load_amazon_data_if_available()
    
    # 3. Analyze dataset
    analyze_dataset(df)
    
    # 4. Save datasets
    save_datasets(df)
    
    print("\n✅ Data preparation completed successfully!")
    print("📁 You can now proceed to train the model.")
    
    # Show first few rows for verification
    print("\n🔍 First 3 rows of processed data:")
    print(df[['review', 'sentiment']].head(3))