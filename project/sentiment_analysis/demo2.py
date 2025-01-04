import re
from collections import defaultdict
import math


# Step 1: Create a dataset
data = [
    ("I love this product, it's amazing!", "Positive"),
    ("This is the worst experience ever.", "Negative"),
    ("Absolutely fantastic and I enjoy using it.", "Positive"),
    ("Not good, I hate it.", "Negative"),
    ("Such a wonderful day to have this!", "Positive"),
    ("Terrible service, very disappointed.", "Negative")
]

# Step 2: Preprocess the text
def preprocess(text):
    # Convert to lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text.split()

# Step 3: Count word frequencies for each class
word_freq = {"Positive": defaultdict(int), "Negative": defaultdict(int)}
class_counts = {"Positive": 0, "Negative": 0}

for text, label in data:
    words = preprocess(text)
    class_counts[label] += 1
    for word in words:
        word_freq[label][word] += 1

# Step 4: Calculate probabilities for Naive Bayes
def train_naive_bayes(word_freq, class_counts):
    vocab = set(word for label in word_freq for word in word_freq[label])
    total_docs = sum(class_counts.values())
    class_probs = {label: class_counts[label] / total_docs for label in class_counts}
    word_probs = {}
    
    for label in word_freq:
        total_words = sum(word_freq[label].values())
        word_probs[label] = {
            word: (word_freq[label][word] + 1) / (total_words + len(vocab))  # Laplace smoothing
            for word in vocab
        }
    
    return class_probs, word_probs, vocab

class_probs, word_probs, vocab = train_naive_bayes(word_freq, class_counts)

# Step 5: Predict the sentiment of a new sentence
def predict(text, class_probs, word_probs, vocab):
    words = preprocess(text)
    scores = {}
    
    for label in class_probs:
        scores[label] = math.log(class_probs[label])  # Use log for numerical stability
        for word in words:
            if word in vocab:
                scores[label] += math.log(word_probs[label].get(word, 1 / (sum(word_freq[label].values()) + len(vocab))))
    
    return max(scores, key=scores.get)

# Step 6: Test the model
test_sentence = "I enjoy this wonderful product!"
prediction = predict(test_sentence, class_probs, word_probs, vocab)
print(f"The sentiment for the sentence '{test_sentence}' is: {prediction}")

# Step 7: Evaluate the model with test data
test_data = [
    ("I hate the bad service.", "Negative"),
    ("What a fantastic experience!", "Positive")
]

correct = 0
for text, label in test_data:
    pred = predict(text, class_probs, word_probs, vocab)
    correct += (pred == label)

accuracy = correct / len(test_data) * 100
print(f"Model Accuracy: {accuracy:.2f}%")
