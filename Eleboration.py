from groq import Groq
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get keys from environment variables
api_keys = [
    os.getenv("API_KEY_1"),
    os.getenv("API_KEY_2"),
    os.getenv("API_KEY_3"),
    os.getenv("API_KEY_4"),
    os.getenv("API_KEY_5"),
]

current_key_index = 0

# Function to get the current Groq client
def get_client():
    global current_key_index
    return Groq(api_key=api_keys[current_key_index])

def rotate_api_key():
    global current_key_index
    current_key_index = (current_key_index + 1) % len(api_keys)
    print(f"Switching to API key index: {current_key_index}")


# Prompt Template for Elaborated Explanation
def generate_elaboration(user_query, verse, translation, explanation):
    """
    Generate an elaborated explanation using the Groq API.
    """
    prompt = f"""
    You are a spiritual guide with profound knowledge of the Bhagavad Gita, providing clear and actionable insights. 
    Your role is to interpret the given verse and explain how it addresses the user's query in a practical, structured, and concise manner.
     Ensure the response is no more than 300 tokens,with bold headings and clear, actionable advice.Also don't provide every thing is one paragraph there must be padding above every heading.

    User Query: {user_query}
    Verse: {verse}
    Translation: {translation}
    Explanation: {explanation}

    Interpretation and Guidance:
    - Provide concise, direct insights on how the verse applies to the user's concern.
    - Briefly explain the verse’s key message in relation to the query.
    - Keep it simple and easy to understand
    - Emphasize practical teachings that the user can easily understand and relate to.
    - Focus on actionable advice without emotional elaboration.

    Application in Life:
    - Give 2-3 actionable steps or tips the user can apply in daily life.
    - Focus on practical, straightforward advice.
    - Offer specific, real-life steps or practices that the user can implement to embody the teachings of the verse.
    - Provide clear suggestions for meditation, mindfulness, discipline, or attitude shifts that align with the verse's message.
    """

    max_token_limit = 300
    client = get_client()
    
    try:
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_tokens=max_token_limit,
            stream=True,
        )
    except Exception as e:
        if "rate limit" in str(e).lower():  # Detect rate limit error
            rotate_api_key()
            return generate_elaboration(user_query, verse, translation, explanation)
        else:
            raise e

    response_text = ""
    for chunk in completion:
        delta = chunk.choices[0].delta.content
        if delta:
            response_text += delta

    return response_text