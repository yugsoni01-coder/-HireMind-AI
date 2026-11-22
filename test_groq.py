
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load the API key
load_dotenv()

# Initialize Groq AI model
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# Ask AI a question
response = llm.invoke(
    "Explain an AI recruitment system in simple words."
)

# Display AI response
print("\nAI RESPONSE:")
print(response.content)
