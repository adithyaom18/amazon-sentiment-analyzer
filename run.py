from app import app

if __name__ == '__main__':
    print("🚀 Starting Amazon Sentiment Analyzer...")
    print("📊 Model: Logistic Regression")
    print("🌐 Web interface: http://localhost:5000")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)