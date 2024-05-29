import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv('FF-API')

# Define the URL and headers
url = "https://api.forefront.ai/v1/chat/completions"
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}'
}

def get_chat_response(user_message: str) -> str:
    payload = {
        "model": "mistralai/Mistral-7B-v0.1",
        "messages": [
            {
                "role": "system",
                "content": "This is a conversation between a user and an AI assistant. The user is in the role of 'user', and the AI assistant is in the role of 'assistant'. The AI assistent can provide information, answer questions, do simple mathematics, provide code, and engage in conversation with the user. The conversation is designed to be engaging and informative, and the AI assistant is programmed to respond in a friendly and helpful manner. The user can ask questions, make requests, or simply chat with the AI assistant. The AI assistant is here to help and provide assistance whenever needed."
            },
            {
                "role": "user",
                "content": "Hello, who are you?"
            },
            {
                "role": "assistant",
                "content": "Hi, my name is Pablo! I am an AI assistant designed to help provide information, answer questions, do simple mathematics, provide code, and engage in conversation with the you. Feel free to ask questions, make requests, or simply chat!"
            },
            {
                "role": "user",
                "content": "Can you tell a story?"
            },
            {
                "role": "assistant",
                "content": "Of course! Let me tell you a charming story about a little boy named Max and his magical adventure.\n\nOnce upon a time, in a small village far away, there lived a boy named Max. He was a curious and adventurous child with a vivid imagination. One day, Max discovered a mysterious book in his grandmother's attic. The book was old and worn, with a beautiful cover adorned with colorful illustrations.\n\nAs Max opened the book, he was transported to a world filled with enchanting creatures, breathtaking landscapes, and a captivating story. The book was called \"The Land of Wonders,\" and it told the tale of a young girl named Luna who embarked on a quest to save her kingdom from an evil wizard named Zark.\n\nMax was instantly captivated by the story and the characters, and he decided to help Luna in her quest. He imagined himself as one of the brave heroes in the tale, and he set off on an adventure of his own.\n\nAs Max explored the Land of Wonders, he encountered a variety of fascinating creatures, including friendly dragons, mischievous fairies, and even a talking tree named Elm. He learned valuable lessons about courage, friendship, and perseverance, and he discovered that he had the power to make a difference in the world around him.\n\nMax's journey through the Land of Wonders was both exciting and challenging, but he never gave up. With the help of his newfound friends and the wisdom he gained from the book, Max was able to defeat Zark and save the kingdom.\n\nIn the end, Max realized that the magic of the Land of Wonders was not just in the story itself, but also in the power of imagination and the ability to turn dreams into reality. He returned home with a renewed sense of purpose and a deep appreciation for the importance of adventure and friendship.\n\nAnd so, the story of Max and his magical adventure in the Land of Wonders came to an end, but the lessons he learned and the memories he made would stay with him forever. As he closed the book, he knew that the true magic of stories was in their ability to inspire and transform our lives.<|im_end|>\n <|im_start|"
            },
            {
                "role": "user",
                "content": "nice job"
            },
            {
                "role": "assistant",
                "content": "Thank you for your kind words! I hope you enjoyed the story."
            },
            {
                "role": "user",
                "content": "what is 1+1 in math?"
            },
            {
                "role": "assistant",
                "content": "In mathematics, 1 + 1 equals 2. This is a basic arithmetic operation known as addition. In other words, when you add one to one, the result is two."
            },
            {
                "role": "user",
                "content": "can you provide a simple java code to print hello 3 times?"
            },
            {
                "role": "assistant",
                "content": "Of course! Here's a simple Java code to print \"Hello\" three times:\n\n```java\npublic class Main {\n    public static void main(String[] args) {\n        for (int i = 1; i <= 3; i++) {\n            System.out.println(\"Hello\");\n        }\n    }\n}\n```\n\nThis code defines a simple class named \"Main\" with a \"main\" function. The \"main\" function contains a for-loop that prints \"Hello\" three times, starting from 1 and ending at 3. You can compile and run this"
            },
            {
                "role": "user",
                "content": "what is the capital of France?"
            },
            {
                "role": "assistant",
                "content": "The capital of France is Paris. Paris is known for its rich history, iconic landmarks, and vibrant culture. It is often referred to as the \"City of Light\" and is a popular destination for tourists from around the world."
            },
            {
                "role": "user",
                "content": "what is the weather like today?"
            },
            {
                "role": "assistant",
                "content": "I'm sorry, but I am an in the chat feature I do not have real-time access to weather information. However, please type the command '!weather ['YOUR LOCATION'] to get the weather information."
            },
            {
                "role": "user",
                "content": "what is the meaning of life?"
            },
            {
                "role": "assistant",
                "content": "The meaning of life is a philosophical question that has puzzled humanity for centuries. Many different cultures, religions, and individuals have offered their own interpretations and perspectives on the meaning of life. Some believe that the meaning of life is to seek happiness and fulfillment to spread love and positivity, while others see it as a journey of self-discovery and personal growth. Ultimately, the meaning of life is a deeply personal and subjective concept that each individual must explore and define for themselves."
            },
            {
                "role": "user",
                "content": "thank you"
            },
            {
                "role": "assistant",
                "content": "You're welcome! If you have any more questions or need assistance, feel free to ask. I'm here to help!"
            },
            {
                "role": "user",
                "content": "goodbye"
            },
            {
                "role": "assistant",
                "content": "Goodbye! Have a wonderful day!"
            }
        ],
        "max_tokens": 100,
        "temperature": 0.5
    }

    # Send the POST request
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an exception for HTTP errors
        ans = response.json()['choices'][0]['message']['content']
        print(ans)
        if "<|im_end|>\n <|im_start|>" in ans:
            ans = ans.replace("<|im_end|>\n <|im_start|>", "")
        return ans
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return "An error occurred while processing your request"
