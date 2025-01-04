import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import kagglehub  # For downloading Kaggle datasets

# Step 1: Download the dataset from Kaggle
path = kagglehub.dataset_download("jp797498e/twitter-entity-sentiment-analysis")
print("Path to dataset files:", path)

# Load the main CSV file (replace 'train.csv' with the actual dataset file name from the downloaded files)
dataset_path = f"{path}/train.csv"  # Adjust file name based on the actual dataset structure
df = pd.read_csv(dataset_path)

# Print dataset info for verification
print("Dataset Loaded:")
print(df.head())

# Step 2: Preprocess the dataset
# Ensure the dataset contains relevant columns for text and sentiment
# Modify 'text_column_name' and 'label_column_name' based on your dataset's structure
text_column_name = 'text'  # Replace with the actual column name for text in your dataset
label_column_name = 'sentiment'  # Replace with the actual column name for labels in your dataset

# If necessary, filter and rename columns
df = df[[text_column_name, label_column_name]]
df.rename(columns={text_column_name: 'text', label_column_name: 'label'}, inplace=True)

# Print dataset structure after preprocessing
print("Preprocessed Dataset:")
print(df.head())

# Step 3: Split the data into training and test sets
texts = df['text']
labels = df['label']
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Step 4: Create a pipeline with CountVectorizer and MultinomialNB
model = Pipeline([
    ('vectorizer', CountVectorizer()),  # Convert text to word count vectors
    ('classifier', MultinomialNB())    # Multinomial Naive Bayes classifier
])

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")

# Step 7: Test a new sentence
test_sentence = "I enjoy this wonderful product!"
prediction = model.predict([test_sentence])[0]
print(f"The sentiment for the sentence '{test_sentence}' is: {prediction}")
