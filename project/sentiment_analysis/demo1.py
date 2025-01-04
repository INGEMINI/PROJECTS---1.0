# Dictionary of positive and negative words
sentiment_dict = {
    'positive': ['happy', 'joy', 'love', 'good', 'great', 'excellent', 'amazing', 'wonderful'],
    'negative': ['sad', 'angry', 'hate', 'bad', 'terrible', 'awful', 'horrible', 'poor']
}

# Function to perform sentiment analysis
def sentiment_analysis(text):
    # Lowercase the input text and split it into words
    words = text.lower().split()
    
    # Initialize counters for positive and negative words
    positive_count = 0
    negative_count = 0
    
    # Check each word in the input text
    for word in words:
        if word in sentiment_dict['positive']:
            positive_count += 1
        elif word in sentiment_dict['negative']:
            negative_count += 1
    
    # Determine overall sentiment
    if positive_count > negative_count:
        return "Positive sentiment"
    elif negative_count > positive_count:
        return "Negative sentiment"
    else:
        return "Neutral sentiment"

# Example usage
text = "I love programming but sometimes it feels terrible"
result = sentiment_analysis(text)
print(result)



