import base64
import requests
import io
import json
import os
import logging
import argparse
from PIL import Image
from dotenv import load_dotenv
from nlp_processor import NLPProcessor
from medical_ner import extract_medical_entities

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# API configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ API KEY is not set in the .env file")

# Initialize NLP processor
nlp = NLPProcessor()

# Define system prompts for different query types
SYSTEM_PROMPTS = {
    "general": """You are an AI medical assistant specialized in analyzing medical images. 
    Provide informative, evidence-based responses while being clear about limitations. 
    Never make definitive diagnoses. Always recommend consulting healthcare professionals.""",
    
    "diagnosis": """You are an AI medical assistant specialized in analyzing medical images for potential diagnoses.
    Analyze the visible symptoms, patterns, and abnormalities in the image.
    Consider differential diagnoses and explain your reasoning.
    Clearly state that this is NOT a definitive diagnosis and always recommend professional medical consultation.""",
    
    "treatment": """You are an AI medical assistant specialized in suggesting potential treatment approaches.
    Based on what's visible in the image, discuss typical treatment options that might be relevant.
    Include general information about medications, procedures, or therapies if applicable.
    Always emphasize that treatment decisions must be made by healthcare professionals.""",
    
    "explain": """You are an AI medical assistant specialized in explaining medical imagery.
    Provide a detailed explanation of the anatomical structures, medical devices, or conditions visible in the image.
    Use medical terminology with clear explanations for non-medical professionals.
    Include educational information about what is shown and its significance."""
}

def analyze_query_semantics(query):
    """
    Analyze query semantics and return suggestions for improvement
    
    Args:
        query: User query
        
    Returns:
        Dictionary with semantic analysis and suggestions
    """
    # Extract entities
    ner_result = extract_medical_entities(query)
    
    # Check for query complexity
    word_count = len(query.split())
    is_complex = word_count > 10
    
    # Get NLP processor
    nlp = NLPProcessor()
    detected_intent = nlp.detect_query_intent(query)
    
    # Generate suggestions
    suggestions = []
    
    if ner_result["entity_count"] == 0:
        suggestions.append("Try adding specific medical terms to improve query precision")
    
    if not is_complex:
        suggestions.append("Consider adding more context or details to your query")
    
    if detected_intent == "general":
        suggestions.append("Using terms like 'diagnose', 'treat', or 'explain' can help focus your query")
    
    # Return analysis
    return {
        "intent": detected_intent,
        "complexity": "complex" if is_complex else "simple",
        "entity_count": ner_result["entity_count"],
        "suggestions": suggestions
    }

def process_image(image_path, query, query_type="general"):
    """
    Process a medical image with an enhanced query
    
    Args:
        image_path: Path to the image file
        query: User's query about the image
        query_type: Type of analysis to perform (general, diagnosis, treatment, explain)
        
    Returns:
        Dictionary containing the response or error
    """
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return {"error": f"Image file not found: {image_path}"}
            
        # Read and encode image
        with open(image_path, "rb") as image_file:
            image_content = image_file.read()
            
        # Validate image format
        try:
            img = Image.open(io.BytesIO(image_content))
            img.verify()
        except Exception as e:
            logger.error(f"Invalid image format: {str(e)}")
            return {"error": f"Invalid image format: {str(e)}"}
            
        # Encode image for API
        encoded_image = base64.b64encode(image_content).decode("utf-8")
        
        # Enhance query using NLP
        enhanced_query = nlp.enhance_query(query, query_type)
        logger.info(f"Enhanced query: '{enhanced_query}'")
        
        # Get system prompt based on query type
        system_prompt = SYSTEM_PROMPTS.get(query_type, SYSTEM_PROMPTS["general"])
        
        # Prepare messages for API request
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": enhanced_query},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]
            }
        ]
        
        # Make API request
        logger.info(f"Sending request to model API...")
        response = make_api_request(messages)
        
        # Process response
        if "error" in response:
            return response
            
        # Store interaction data for potential learning
        store_interaction(query, enhanced_query, query_type, response["content"])
        
        return response
        
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        return {"error": f"An unexpected error occurred: {str(e)}"}

def make_api_request(messages):
    """
    Make a request to the Groq API
    
    Args:
        messages: List of message objects for the API request
        
    Returns:
        Dictionary containing the response content or error
    """
    try:
        # Send request to API
        response = requests.post(
            GROQ_API_URL,
            json={
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": messages,
                "max_tokens": 1500,
                "temperature": 0.2  # Lower temperature for more factual responses
            },
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            timeout=30
        )
        
        # Process response
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            logger.info(f"Received response from API: {content[:100]}...")
            return {"content": content}
        else:
            error_msg = f"Error from model API: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return {"error": error_msg}
            
    except requests.exceptions.Timeout:
        logger.error("Request timed out")
        return {"error": "Request to language model timed out"}
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return {"error": f"Request error: {str(e)}"}
        
    except Exception as e:
        logger.error(f"Unexpected error in API request: {str(e)}")
        return {"error": f"Unexpected error in API request: {str(e)}"}

def store_interaction(original_query, enhanced_query, query_type, response):
    """
    Store the interaction data for potential future learning
    
    Args:
        original_query: Original user query
        enhanced_query: Enhanced query sent to API
        query_type: Type of query (general, diagnosis, treatment, explain)
        response: Model response
    """
    try:
        # Create data directory if it doesn't exist
        os.makedirs("data/interactions", exist_ok=True)
        
        # Create interaction data object
        interaction = {
            "timestamp": f"{import_time().time()}",
            "original_query": original_query,
            "enhanced_query": enhanced_query,
            "query_type": query_type,
            "response": response
        }
        
        # Generate filename with timestamp
        filename = f"data/interactions/interaction_{int(import_time().time())}.json"
        
        # Write to file
        with open(filename, "w") as f:
            json.dump(interaction, f, indent=2)
            
        logger.info(f"Stored interaction data to {filename}")
        
    except Exception as e:
        logger.error(f"Failed to store interaction: {str(e)}")

def import_time():
    """Import time module lazily to avoid circular imports"""
    import time
    return time

def main():
    """Run as a command-line application"""
    parser = argparse.ArgumentParser(description="Medical Image Analysis Tool")
    parser.add_argument("image_path", help="Path to the medical image file")
    parser.add_argument("query", help="Query about the medical image")
    parser.add_argument("--type", choices=["general", "diagnosis", "treatment", "explain", "auto"], 
                        default="auto", help="Type of analysis to perform")
    parser.add_argument("--analyze-only", action="store_true", 
                        help="Only analyze the query without processing image")
    
    args = parser.parse_args()
    
    # If analyze-only flag is set, just analyze the query
    if args.analyze_only:
        analysis = analyze_query_semantics(args.query)
        print("\n=== QUERY ANALYSIS ===\n")
        print(f"Detected intent: {analysis['intent']}")
        print(f"Query complexity: {analysis['complexity']}")
        print(f"Entity count: {analysis['entity_count']}")
        print("\nSuggestions:")
        for suggestion in analysis['suggestions']:
            print(f"- {suggestion}")
        print("\n=====================\n")
        return
    
    # Process the image
    # If auto-detect is enabled, detect query type
    if args.type == "auto":
        nlp = NLPProcessor()
        query_type = nlp.detect_query_intent(args.query)
        print(f"Auto-detected query type: {query_type}")
    else:
        query_type = args.type
    
    result = process_image(args.image_path, args.query, query_type)
    
    # Print the result
    if "error" in result:
        print(f"ERROR: {result['error']}")
    else:
        print("\n=== ANALYSIS RESULT ===\n")
        print(result["content"])
        print("\n======================\n")

if __name__ == "__main__":
    main()