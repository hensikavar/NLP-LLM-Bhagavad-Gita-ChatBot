import random
from flask import Flask, request, jsonify
import json
from flask_cors import CORS
from Eleboration import generate_elaboration
from rapidfuzz import process
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load data
with open('./alldata.json', "r", encoding="utf-8-sig") as f:
    content = f.read().strip()  # Remove BOM and extra spaces
    verses = json.loads(content)

# Preprocess verses
translations = [verse['translation'] for verse in verses]
explanations = [verse['explanation'] for verse in verses]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(translations)

# Initialize semantic model
semantic_model = SentenceTransformer('all-mpnet-base-v2')
embeddings = semantic_model.encode(translations)

# Flask app
app = Flask(__name__)
CORS(app, origins="*")

# List of friendly responses for casual queries
friendly_responses = {
    "hi": [
        "Hello! How can I help you today with insights from the Bhagavad Gita?",
        "Hi there! Feel free to ask me a question about life, spirituality, or any challenges you're facing."
    ],
    "hello": [
        "Hello! How can I help you today with insights from the Bhagavad Gita?",
        "Hi there! Feel free to ask me a question about life, spirituality, or any challenges you're facing."
    ],
    "how are you": [
        "I’m here and ready to assist! What wisdom from the Bhagavad Gita can I share with you today?",
        "I'm here to help you explore the teachings of the Bhagavad Gita. What can I do for you?"
    ],
    "what's up": [
        "Just here, ready to share wisdom! What life challenge or question would you like guidance on today?",
        "I’m here to help you discover actionable insights from the Gita. What would you like to know?"
    ],
    # "tell me something": [
    #     "Did you know the Bhagavad Gita emphasizes inner peace and duty (dharma) as keys to a fulfilled life? Let me know if you'd like to explore more!",
    #     "Here's something to ponder: The Gita teaches that true success lies in effort, not in the results. How can I assist you further with this wisdom?"
    # ],
    "repeated casual queries": [
        "It’s always good to connect! If you have a question about life or need guidance, let me know how I can help!",
        "I’m here to help you reflect on life’s deeper questions. What’s on your mind today?"
    ],
    "good morning": [
        "Good morning! How can I assist you with wisdom from the Bhagavad Gita today?",
        "Good morning! Ready to dive into the teachings of the Gita. What would you like to explore?"
    ],
    "good afternoon": [
        "Good afternoon! How can I help you today with insights from the Bhagavad Gita?",
        "Good afternoon! Let me know if you have any questions about the teachings of the Gita."
    ],
    "good evening": [
        "Good evening! How can I help you today with insights from the Bhagavad Gita?",
        "Good evening! Let me know if you have any questions about the teachings of the Gita."
    ],
    "good night": [
        "Good night! May peace and wisdom from the Bhagavad Gita guide you in your dreams.",
        "Good night! Sleep well and let the teachings of the Gita provide you with peace and tranquility."
    ],
    "sorry": [
        "No need to apologize! I’m here to assist you with wisdom from the Bhagavad Gita.",
        "Apologies are unnecessary! I'm happy to guide you with the teachings of the Bhagavad Gita."
    ],
    "please": [
        "You're welcome! Feel free to ask any questions about the Bhagavad Gita.",
        "It's my pleasure to help! How can I assist you with the wisdom of the Bhagavad Gita today?"
    ],
    "love you": [
        "Love and peace to you too! 😊 How can I guide you with the wisdom of the Bhagavad Gita?",
        "Thank you! Love and kindness are central to the teachings of the Gita. How can I help you today?"
    ]
}

# Initialize a variable to track user input history (e.g., in memory, or session)
user_history = []

def get_friendly_response(user_query):
    # Normalize user input
    user_query = user_query.lower().strip()

    # Fuzzy match the input with predefined friendly responses
    match = process.extractOne(user_query, list(friendly_responses.keys()))
    if match:
        best_match, score = match[:2]  
        if score > 70:  # Threshold for considering it a match
            # Check if the user query was already asked
            if user_query in user_history:
                return random.choice(friendly_responses["repeated casual queries"])

            # Add to history for future tracking
            user_history.append(user_query)

            # Return an appropriate response based on the best match
            return random.choice(friendly_responses[best_match])




    # Default response if no match is found
    return "I’m here to help you explore the teachings of the Bhagavad Gita. Please feel free to ask any question!"

# Home route
@app.route('/')
def home():
    return 'Hello'

# Get response for the user input
@app.route('/get_response', methods=['POST'])
def get_response():
    user_query = request.json.get('user_input')
    if not user_query or not isinstance(user_query, str):
        return jsonify({"error": "Invalid input. Please provide a valid string in 'user_input'"}), 400

    # Check for friendly/casual queries first
    friendly_reply = get_friendly_response(user_query)
    if friendly_reply != "I’m here to help you explore the teachings of the Bhagavad Gita. Please feel free to ask any question!":
        return jsonify({"response": friendly_reply})

    # TF-IDF matching
    query_vec = vectorizer.transform([user_query])
    similarities = cosine_similarity(X, query_vec).flatten()
    best_match_idx = similarities.argmax()

    # Semantic search
    query_embedding = semantic_model.encode([user_query])
    semantic_scores = cosine_similarity(query_embedding, embeddings).flatten()
    best_semantic_match_idx = semantic_scores.argmax()

    # Select best match
    best_match = verses[best_semantic_match_idx]

    # Generate elaborated response
    verse = best_match.get("verse_number", "N/A")
    translation = best_match.get("translation", "N/A")
    explanation = best_match.get("explanation", "N/A")
    elaborated_response = generate_elaboration(user_query, verse, translation, explanation)

    # Build response
    response = {
        "verse_number": verse,
        "translation": translation,
        "explanation": explanation,
        "problem_category": best_match.get("problem_category", "N/A"),
        "elaboration": elaborated_response
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run()