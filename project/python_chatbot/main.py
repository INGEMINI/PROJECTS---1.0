import openai

# Set your OpenAI API key
openai.api_key = "my api key "

def chat_with_gpt(prompt):
    try:
        # Send the user prompt to OpenAI Chat API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract the chatbot's reply from the response
        return response['choices'][0]['message']['content'].strip()

    except Exception as e:
        # Handle exceptions and provide error feedback
        return f"An error occurred: {e}"

if __name__ == "__main__":
    print("Chatbot: Hi! I'm here to chat with you. Type 'quit', 'exit', or 'bye' to leave the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Chatbot: Goodbye! Have a great day!")
            break
        response = chat_with_gpt(user_input)
        print(f"Chatbot: {response}")
