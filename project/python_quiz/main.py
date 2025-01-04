questions = {
    "What is the capital of India?": "Delhi",
    "What is 2 + 2?": "4",
    "What is the color of the sky?": "blue"
}

for question, answer in questions.items():
    user_answer = input(question + " ")
    if user_answer.lower() == answer.lower():
        print("Correct!")
    else:
        print(f"Wrong! The correct answer is {answer}.")
