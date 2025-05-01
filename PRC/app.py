from flask import Flask, render_template, request
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords, words
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('words', quiet=True)

# Get English words set
english_words = set(words.words())

# Sentiment indicators
POSITIVE_KEYWORDS = {
    'excellent', 'awesome', 'great', 'fantastic', 'perfect', 'love','like',
    'wonderful', 'amazing', 'superb', 'outstanding', 'best', 'favorite',
    'recommend', 'impressive', 'brilliant', 'happy', 'pleased', 'satisfied',
    'high quality', 'good', 'nice', 'worth it', 'exceeds expectations'
}

NEUTRAL_KEYWORDS = {
    'okay', 'average', 'moderate', 'neutral', 'decent', 'adequate',
    'mediocre', 'fair', 'so-so', 'acceptable', 'tolerable', 'not bad',
    'not good not bad', 'neither good nor bad', 'middle', 'middling',
    'alright', 'ordinary', 'passable', 'standard', 'typical'
}

NEGATIVE_KEYWORDS = {
    'terrible', 'awful', 'horrible', 'bad', 'poor', 'disappointing',
    'worst', 'waste', 'rubbish', 'garbage', 'broken', 'defective',
    'fail', 'faulty', 'useless', 'not good', 'not worth', 'regret',
    'dislike', 'hate', 'annoying', 'problem', 'issue', 'damaged','not working',
    'return', 'refund', 'complaint', 'junk', 'unhappy', 'frustrated'
}

# Load and prepare data
try:
    df = pd.read_csv("final_review_set.csv")
except:
    data = {
        'cleaned_review': [
            'excellent product, love it!', 
            'terrible experience, would not buy again',
            'it was okay, nothing special',
            'amazing quality, exceeds expectations',
            'moderate performance for the price',
            'neutral feeling about this product',
            'not good not bad, just average',
            'decent but could be better',
            'awful customer service',
            'broken upon arrival',
            'mediocre at best',
            'best purchase ever',
            'highly recommend this product',
            'worst experience with this company',
            'perfect condition, very happy',
            'not worth the money',
            'good value for the price',
            'frustrating to use'
        ],
        'review_score': [5, 1, 3, 5, 3, 3, 3, 3, 1, 1, 2, 5, 5, 1, 5, 1, 4, 2]
    }
    df = pd.DataFrame(data)

# Enhanced text cleaning preserving sentiment cues
def clean_text(text):
    text = str(text).lower()
    return re.sub(r"[^\w\s!?]", '', text)

df['cleaned_text'] = df['cleaned_review'].apply(clean_text)

# Check if input contains meaningful words
def is_valid_sentence(text):
    words_list = text.split()
    valid_word_count = sum(1 for word in words_list if word in english_words)
    return valid_word_count > 0  # At least one valid English word

# Comprehensive sentiment mapping
def get_sentiment(score, text):
    text = clean_text(text)
    
    if any(keyword in text for keyword in POSITIVE_KEYWORDS):
        return 'Positive'
    if any(keyword in text for keyword in NEGATIVE_KEYWORDS):
        return 'Negative'
    if any(keyword in text for keyword in NEUTRAL_KEYWORDS):
        return 'Neutral'
    return 'Positive' if score > 3 else 'Negative' if score < 3 else 'Neutral'

df['sentiment'] = df.apply(lambda row: get_sentiment(row['review_score'], row['cleaned_review']), axis=1)

# Enhanced model training
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 3),
    max_features=10000,
    min_df=2
)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['sentiment']

# Model configuration
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
    solver='liblinear',
    C=0.5
)
model.fit(X, y)

# Prediction function
def predict(text):
    cleaned = clean_text(text)
    
    # Validate the sentence first
    if not is_valid_sentence(cleaned):
        return None, None
    
    try:
        if any(pos in cleaned for pos in POSITIVE_KEYWORDS):
            return 'Positive', {'Positive': 0.9, 'Neutral': 0.05, 'Negative': 0.05}
        if any(neg in cleaned for neg in NEGATIVE_KEYWORDS):
            return 'Negative', {'Positive': 0.05, 'Neutral': 0.05, 'Negative': 0.9}
        if any(neutral in cleaned for neutral in NEUTRAL_KEYWORDS):
            return 'Neutral', {'Positive': 0.1, 'Neutral': 0.8, 'Negative': 0.1}
        
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        return pred, dict(zip(model.classes_, proba))
    except:
        return 'Neutral', {'Positive': 0.33, 'Neutral': 0.34, 'Negative': 0.33}

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    user_input = request.form.get('review', '')
    if not user_input.strip():
        return render_template('error.html', message="Please enter some text")
    
    prediction, probabilities = predict(user_input)
    
    if prediction is None:  # Invalid input case
        return render_template('result.html', 
                             review=user_input,
                             error="Please enter a valid sentence with proper meaning")
    
    return render_template('result.html',
                         review=user_input,
                         result=prediction,
                         probabilities=probabilities)

@app.route('/error')
def error():
    return render_template('error.html')

if __name__ == '__main__':
    app.run(debug=True)