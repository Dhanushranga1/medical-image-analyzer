from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import base64
import requests
import io
from PIL import Image
from dotenv import load_dotenv
import os
import logging
import json
from typing import Optional
import time
from medical_ner import extract_medical_entities, extract_medical_relationships
from nlp_processor import NLPProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# API configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file")

# Define the directory to store past queries (for potential fine-tuning)
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

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

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze_text")
async def analyze_text(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        # Extract entities and relationships
        entities = extract_medical_entities(text)
        relationships = extract_medical_relationships(text)
        
        # Detect intent using NLP processor
        nlp_processor = NLPProcessor()
        intent = nlp_processor.detect_query_intent(text)
        
        return JSONResponse(status_code=200, content={
            "entities": entities,
            "relationships": relationships,
            "detected_intent": intent
        })
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing text: {str(e)}")

@app.post("/upload_and_query")
async def upload_and_query(
    image: UploadFile = File(...),
    query: str = Form(...),
    original_query: Optional[str] = Form(None),
    query_type: str = Form("general")
):
    try:
        # Process image
        image_content = await image.read()
        if not image_content:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Validate image
        try:
            img = Image.open(io.BytesIO(image_content))
            img.verify()
        except Exception as e:
            logger.error(f"Invalid image format: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
        
        # Encode image for API request
        encoded_image = base64.b64encode(image_content).decode("utf-8")
        
        # Enhance NLP processing
        nlp_processor = NLPProcessor()
        
        # If query_type is "auto", detect it automatically
        if query_type == "auto":
            query_type = nlp_processor.detect_query_intent(query)
            logger.info(f"Auto-detected query type: {query_type}")
        
        # Get enhanced query
        enhanced_query = nlp_processor.enhance_query(query, query_type)
        
        # Extract medical entities
        ner_result = extract_medical_entities(enhanced_query)
        relationships = extract_medical_relationships(query)
        
        # Get appropriate system prompt based on query type
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
        try:
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
            
            # Handle API response
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"]
                
                logger.info(f"Processed response: {answer[:100]}...")
                
                # Store query and response for potential training
                store_interaction(
                    query_type=query_type,
                    original_query=original_query or query,
                    enhanced_query=enhanced_query,
                    response=answer,
                    model="meta-llama/llama-4-scout-17b-16e-instruct"
                )
                
                # Return response with NLP results
                return JSONResponse(status_code=200, content={
                    "response": answer,
                    "entities": ner_result,
                    "relationships": relationships,
                    "detected_intent": query_type,
                    "enhanced_query": enhanced_query
                })
            else:
                error_msg = f"API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise HTTPException(status_code=response.status_code, detail=error_msg)
                
        except requests.exceptions.Timeout:
            logger.error("Request timed out")
            raise HTTPException(status_code=504, detail="Request to language model timed out")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")
            
    except HTTPException as he:
        logger.error(f"HTTP Exception: {str(he)}")
        raise he
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

def store_interaction(query_type, original_query, enhanced_query, response, model):
    """Store the interaction data for potential future use in fine-tuning"""
    try:
        data = {
            "timestamp": time.time(),
            "query_type": query_type,
            "original_query": original_query,
            "enhanced_query": enhanced_query,
            "response": response,
            "model": model
        }
        
        # Create a unique filename
        filename = f"{DATA_DIR}/interaction_{int(time.time())}.json"
        
        # Save to file
        with open(filename, "w") as f:
            json.dump(data, f)
            
        logger.info(f"Saved interaction data to {filename}")
    except Exception as e:
        logger.error(f"Failed to store interaction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)