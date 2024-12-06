from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from pinecone import Pinecone
from groq import Groq
import json
import os


load_dotenv()

app = Flask(__name__)

INDEX_NAME = "stocks"
NAMESPACE = "stock-descriptions"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    
    """
    Generates embeddings for the given text using a specified Hugging Face model.

    Args:
        text (str): The input text to generate embeddings for.
        model_name (str): The name of the Hugging Face model to use.
                          Defaults to "sentence-transformers/all-mpnet-base-v2".

    Returns:
        np.ndarray: The generated embeddings as a NumPy array.
    """

    model = SentenceTransformer(model_name)
    return model.encode(text).tolist()

def get_augmented_context(formatted_response):

    pinecone_index = pinecone_client.Index(INDEX_NAME)

    filter = formatted_response["filter"]
    print(filter)
    question = formatted_response["question"]

    question_embeddings = get_huggingface_embeddings(question)

    top_matches = pinecone_index.query(
        namespace=NAMESPACE,
        vector=question_embeddings,
        filter=filter,
        top_k=10,
        include_metadata=True
    )

    print(top_matches)

    context = [item['metadata']['text'] for item in top_matches['matches']]
    augmented_context = "<CONTEXT>\n" + "\n\n-------\n\n".join(context[ : 10]) + "\n-------\n</CONTEXT>\n\n\n\n"

    return augmented_context

def get_question(formatted_response):

    question = formatted_response["question"]
    augmented_question = f"""
    MY QUESTION:

    Using ONLY the context provided, answer the following question:
    {question}

    You should mention all the companies that are mentioned in the context. In your answer, I expect to see at least the company name, ticker, and the reason why you think it is relevant to the question.

    Never make reference to the context in your answer. For example, do not say "Based on the context provided, I can tell you that..."

    If there is not enough information in the context to answer the question, you should say "My apologies, but I do not have enough information to answer that question."
    
    """

    return augmented_question

def generate_prompt(formatted_response):

    augmented_context = get_augmented_context(formatted_response)
    augmented_question = get_question(formatted_response)

    augmented_query = augmented_context + augmented_question

    return augmented_query

@app.route("/explore", methods=["POST"])
def explore():

    data = request.json
    user_query = data["user_query"]
    
    prompt = f"""
    You are a pinecone vector database expert. I have indexed stock data into the vector database in the following structure:

    metadata: {{
        "Analyst Recommendation": "string"
        "Business Summary": "string",
        "City": "string",
        "Country": "string",
        "Industry": "string",
        "Market Cap": "number",
        "Name": "string",
        "PE Ratio": "number",
        "Price": "number",
        "Sector": "string",
        "State": "string",
        "Ticker": "string",
        "Volume": "number",
    }}

    For "Analyst Recommendation", the values can be "buy", "hold", or "sell".
    For "Sector", the values can be "Communication Services", "Consumer Discretionary", "Consumer Staples", "Energy", "Financials", "Healthcare", "Industrials", "Technology", "Materials", "Real Estate", "Utilities".
    For "State", the values are the 2-letter codes for the states in the United States.
    For "City", the values are the names of cities.
    For "Country", the values are the names of countries.

    Now given the following user query:

    <USER QUERY>
    {user_query}
    </USER QUERY>

    Extract the information from the user query that can be used to filter the vector database by metadata. 

    I want you to respond in the following JSON format:

    <JSON FORMAT>
    {{
        "filter": {{
            "Market Cap": {{
                "$gte": 10_000_000_000
            }},
        }},
        "question": <the rest of the user query that cannot be used to filter the vector database using metadata>
    }}

    OR

    {{
        "filter": {{
            "$and": [
                {{
                    "Market Cap": {{
                        "$gte": 10_000_000_000
                    }}, 
                }},
                {{
                    "Sector": {{
                        "$eq": "Technology"
                    }},
                }},
            ]
        }},
        "question": <the rest of the user query that cannot be used to filter the vector database using metadata. It should be a question that can be used to query the vector database based on embeddings>
    }}
    </JSON FORMAT>

    Note that for the filter you should only use the following operators:

    Filter	        Example	                                                Description
    $eq	            {{"genre": {{"$eq": "documentary"}}}}	                Matches vectors with the genre “documentary”.
    $ne	            {{"genre": {{"$ne": "drama"}}}}	                        Matches vectors with a genre other than “drama”.
    $gt	            {{"year": {{"$gt": 2019}}}}	                            Matches vectors with a year greater than 2019.
    $gte	        {{"year": {{"$gte": 2020}}}}	                        Matches vectors with a year greater than or equal to 2020.
    $lt	            {{"year": {{"$lt": 2020}}}}	                            Matches vectors with a year less than 2020.
    $lte	        {{"year": {{"$lte": 2020}}}}	                        Matches vectors with a year less than or equal to 2020.
    $in	            {{"genre": {{"$in": ["comedy", "documentary"]}}}}	    Matches vectors with the genre “comedy” or “documentary”.
    $nin	        {{"genre": {{"$nin": ["comedy", "documentary"]}}}}	    Matches vectors with a genre other than “comedy” or “documentary”.
    $exists	        {{"genre": {{"$exists": true}}}}	                    Matches vectors with the “genre” field.

    Operator	    Example	                                                                        Description
    $and	        {{"$and": [{{"genre": {{"$eq": "drama"}}, {{"year": {{"$gte": 2020"}}}}]}}	    Matches vectors with the genre “drama” and a year greater than or equal to 2020.
    $or	            {{"$or": [{{"genre": {{"$eq": "drama"}}, {{"year": {{"$gte": 2020"}}}}]}}	    Matches vectors with the genre “drama” or a year greater than or equal to 2020.

    If you aren't sure about the filter, you should not include it in the JSON. 

    Please consider the all the above information when responding, and take your time in understanding the user query. If necessary break down the problem into smaller steps.
    """

    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    formatted_response = json.loads(response.choices[0].message.content)

    prompt = generate_prompt(formatted_response)

    return jsonify({"prompt": prompt}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
