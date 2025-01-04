import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset from a local CSV file after downloading from Kaggle
dataset_path = "C:/Users/MAHI AGARWAL/Videos/tech/vs codes/project/sentiment_analysis/sentiments/IMDB Dataset.csv"  # Replace with your local file path
df = pd.read_csv(dataset_path)

# Print dataset info for verification
print("Dataset Loaded:")
print(df.head())

# Ensure the dataset contains only 'review' and 'sentiment' columns
df = df[['review', 'sentiment']]  # Adjusting to the correct column names

# Step 2: Split the data into training and test sets
texts = df['review']  # 'review' is the column with the text
labels = df['sentiment']  # 'sentiment' is the column with the labels
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Step 3: Create a pipeline with CountVectorizer and MultinomialNB
model = Pipeline([
    ('vectorizer', TfidfVectorizer()),  # Convert text to word count vectors
    ('classifier', MultinomialNB())    # Multinomial Naive Bayes classifier
])

# Step 4: Train the model
model.fit(X_train, y_train)

# Step 5: Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")

# Step 6: Test a new sentence
test_sentence = "I enjoy this wonderful product!"
prediction = model.predict([test_sentence])[0]
print(f"The sentiment for the sentence '{test_sentence}' is: {prediction}")