# amazon_product_review_classification
Sentiment Analysis Web App:-
This is a Flask web application that analyzes customer product reviews and predicts the sentiment as Positive, Neutral, or Negative using a Logistic Regression model trained on TF-IDF features. It also uses keyword-based heuristics to enhance prediction accuracy.

Features:-
Accepts user input for a product review

Predicts sentiment using a hybrid of machine learning and keyword heuristics

Displays sentiment classification along with prediction confidence

Handles missing or invalid input with proper feedback

Automatically falls back on a built-in dataset if the CSV is missing

Tech Stack:-
Frontend: HTML with Flask templating

Backend: Python, Flask

NLP & ML: NLTK, Scikit-learn, TF-IDF Vectorization, Logistic Regression

Installation:-
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/sentiment-analysis-flask.git
cd sentiment-analysis-flask
Create a virtual environment (optional but recommended):

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
(Optional) Add your dataset:

If you have a dataset, place it in the project root and name it final_review_set.csv with columns:

cleaned_review

review_score

Run the app:

bash
Copy
Edit
python app.py
Open in your browser:

Visit http://127.0.0.1:5000

Usage:-
Enter a product review in the text field.

Submit the form.

View the predicted sentiment and confidence levels.

Example Keywords Used:-
Positive: excellent, perfect, love, superb, recommend, etc.

Neutral: okay, average, decent, typical, etc.

Negative: terrible, awful, broken, refund, worst, etc.

File Structure:-
csharp
Copy
Edit
.
├── app.py                  # Main application logic
├── templates/              # HTML templates (index.html, result.html, error.html)
├── final_review_set.csv    # (Optional) CSV dataset
├── requirements.txt        # Dependencies list
└── README.md               # Project description
License:-
This project is open source and available under the MIT License.

User Interface:-
The application features a clean, modern, and responsive design using custom CSS and Poppins fonts. It is optimized for both desktop and mobile users.

index.html (Home Page):-
Purpose: Allows users to input a product review for sentiment analysis.

Features:

A styled textarea for review input

A submit button (Analyze Sentiment)

Sample review suggestions for user guidance

Particle animation background using particles.js

Design Highlights:

Elegant card-style layout with hover effects

Theme uses green (positive), blue, and dark accents

Fully responsive for smaller screens

result.html (Result Page):-
Purpose: Displays the sentiment prediction and confidence after form submission.

Features:

Shows the user's input review

Displays sentiment (Positive, Neutral, Negative) with color-coded background

Provides confidence score and detailed probability breakdown

Includes a button to return and analyze another review

Error Handling:

If the input is invalid, an error message is shown in red

Design Highlights:

Consistent card layout with themed styling

Uses sentiment-based color cues:

Green for Positive

Yellow/Orange for Neutral

Red for Negative

Adaptable layout for mobile
