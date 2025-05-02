import re
import string
import logging
from typing import Dict, List, Optional, Tuple
import json
import os
from collections import Counter
from medical_ner import extract_medical_entities


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLPProcessor:
    """NLP utilities for enhancing medical image queries"""
    
    # Medical terminology categories for pattern recognition
    MEDICAL_CATEGORIES = {
        "anatomy": ["heart", "lung", "brain", "liver", "kidney", "bone", "joint", "artery", "vein", 
                   "spine", "skull", "thorax", "abdomen", "pelvis", "extremity", "tissue"],
        
        "imaging": ["x-ray", "xray", "mri", "ct scan", "ultrasound", "radiograph", "sonogram", 
                   "tomography", "pet scan", "mammogram", "fluoroscopy", "angiogram"],
        
        "conditions": ["fracture", "tumor", "lesion", "infection", "inflammation", "pneumonia", 
                      "cancer", "stroke", "clot", "cyst", "nodule", "mass", "calcification"],
        
        "visual_patterns": ["dark", "bright", "shadow", "opacity", "density", "contrast", "signal", 
                           "intensity", "enhancement", "hyperintense", "hypointense", "radiolucent"]
    }
    
    # Template phrases to enhance queries
    ENHANCEMENT_TEMPLATES = {
        "general": [
            "Can you analyze this medical image and {query}?",
            "Looking at this medical image, {query}?",
            "Based on this medical image, please {query}"
        ],
        
        "diagnosis": [
            "What possible conditions could explain what's shown in this {image_type}? {query}",
            "Based on the patterns visible in this medical image, what are potential diagnoses? {query}",
            "What differential diagnoses would you consider for this {image_type}? {query}"
        ],
        
        "treatment": [
            "If this image shows {condition}, what treatment approaches might be appropriate? {query}",
            "What treatment options would typically be considered for the condition shown? {query}",
            "Based on what's visible in this {image_type}, what treatment strategies might a doctor consider? {query}"
        ],
        
        "explain": [
            "Please identify and explain the key structures visible in this {image_type}. {query}",
            "What anatomical features can be identified in this medical image? {query}",
            "Explain what we're seeing in this {image_type}, including any abnormalities. {query}"
        ]
    }
    
    # Knowledge base for common medical terms (expandable)
    KNOWLEDGE_BASE_FILE = "data/medical_knowledge.json"
    
    def __init__(self):
        """Initialize the NLP processor and load knowledge base if available"""
        self.knowledge_base = self._load_knowledge_base()
        self.query_history = []
        
    def _load_knowledge_base(self) -> Dict:
        """Load medical knowledge base from JSON file if available"""
        try:
            if os.path.exists(self.KNOWLEDGE_BASE_FILE):
                with open(self.KNOWLEDGE_BASE_FILE, 'r') as f:
                    return json.load(f)
            else:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.KNOWLEDGE_BASE_FILE), exist_ok=True)
                
                # Create initial knowledge base
                initial_kb = {
                    "imaging_types": {
                        "x-ray": "Radiographic imaging using X-radiation",
                        "mri": "Magnetic Resonance Imaging",
                        "ct": "Computed Tomography scan",
                        "ultrasound": "Imaging using sound waves"
                    },
                    "common_conditions": {
                        "fracture": "Break in bone continuity",
                        "pneumonia": "Inflammation of lung tissue",
                        "tumor": "Abnormal mass of tissue"
                    }
                }
                
                # Save initial knowledge base
                with open(self.KNOWLEDGE_BASE_FILE, 'w') as f:
                    json.dump(initial_kb, f, indent=2)
                    
                return initial_kb
                
        except Exception as e:
            logger.error(f"Error loading knowledge base: {str(e)}")
            return {}
    
    def detect_query_intent(self, query: str) -> str:
        """Detect the intent of a medical query."""
        # Normalize query
        normalized_query = self._normalize_text(query)
        
        # Get NER entities
        ner_result = extract_medical_entities(normalized_query)
        
        # Check for diagnostic intent
        diagnostic_terms = ["diagnose", "diagnosis", "what is", "what could", "identify", "possible condition"]
        if any(term in normalized_query for term in diagnostic_terms):
            return "diagnosis"
        
        # Check for treatment intent
        treatment_terms = ["treat", "treatment", "therapy", "management", "handle", "care for"]
        if any(term in normalized_query for term in treatment_terms):
            return "treatment"
        
        # Check for explanation intent
        explanation_terms = ["explain", "description", "what does", "show", "tell me about", "details"]
        if any(term in normalized_query for term in explanation_terms):
            return "explain"
        
        # Count entity types to determine intent
        entity_types = [entity["label"] for entity in ner_result["entities"]]
        if "DISEASE" in entity_types and entity_types.count("DISEASE") > 0:
            return "diagnosis"
        if "CHEMICAL" in entity_types and entity_types.count("CHEMICAL") > 0:
            return "treatment"
        if "ANATOMY" in entity_types and entity_types.count("ANATOMY") > 0:
            return "explain"
        
        # Default to general
        return "general"

    def generate_semantic_hints(self, ner_result):
        """Generate semantic hints for the query based on NER results"""
        hints = []
        
        # Add hints based on entity types
        entity_types = [entity["label"] for entity in ner_result["entities"]]
        
        if "DISEASE" in entity_types:
            hints.append("Focus on identifying signs of disease in the image")
        
        if "CHEMICAL" in entity_types:
            hints.append("Consider treatment implications visible in the image")
        
        if "ANATOMY" in entity_types:
            hints.append("Highlight the relevant anatomical structures")
        
        if ner_result["has_uncertainty"]:
            hints.append("Address the uncertainty in the query by considering multiple possibilities")
        
        return hints

    def enhance_query(self, query: str, query_type: str = None) -> str:
        """Enhanced version that uses NER and intent detection"""
        # Store original query for learning
        self.query_history.append(query)
        
        # Normalize text
        normalized_query = self._normalize_text(query)
        
        # Extract entities using medical NER
        ner_result = extract_medical_entities(normalized_query)
        
        # Log entities
        logger.info(f"NER Results: {ner_result}")
        
        # Detect intent if not provided
        if not query_type:
            query_type = self.detect_query_intent(normalized_query)
            logger.info(f"Detected intent: {query_type}")
        
        # Generate semantic hints
        semantic_hints = self.generate_semantic_hints(ner_result)
        
        # Extract key terms and other info as before
        medical_terms = self._extract_medical_terms(normalized_query)
        image_type = self._detect_image_type(normalized_query)
        condition = self._detect_medical_condition(normalized_query)
        
        # Select and fill template as before
        if query_type in self.ENHANCEMENT_TEMPLATES:
            templates = self.ENHANCEMENT_TEMPLATES[query_type]
            
            if query_type == "treatment" and condition:
                template = templates[0]
            elif image_type:
                template = next((t for t in templates if "{image_type}" in t), templates[0])
            else:
                template = templates[0]
            
            enhanced_query = template.format(
                query=query,
                image_type=image_type or "medical image",
                condition=condition or "the condition"
            )
        else:
            enhanced_query = f"Analyze this medical image and {query}"
        
        # Add semantic hints
        if semantic_hints:
            enhanced_query += f" {' '.join(semantic_hints)}"
        
        logger.info(f"Enhanced query: {enhanced_query}")
        return enhanced_query
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text by converting to lowercase and removing punctuation"""
        text = text.lower()
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _extract_medical_terms(self, text: str) -> Dict[str, List[str]]:
        """Extract medical terminology from text"""
        found_terms = {category: [] for category in self.MEDICAL_CATEGORIES}
        
        for category, terms in self.MEDICAL_CATEGORIES.items():
            for term in terms:
                if f" {term} " in f" {text} " or text.startswith(f"{term} ") or text.endswith(f" {term}") or text == term:
                    found_terms[category].append(term)
        
        return found_terms
    
    def _detect_image_type(self, text: str) -> Optional[str]:
        """Detect the type of medical image being discussed"""
        for img_type in self.MEDICAL_CATEGORIES["imaging"]:
            if f" {img_type} " in f" {text} " or text.startswith(f"{img_type} ") or text.endswith(f" {img_type}") or text == img_type:
                return img_type
        return None
    
    def _detect_medical_condition(self, text: str) -> Optional[str]:
        """Detect if a specific medical condition is mentioned"""
        for condition in self.MEDICAL_CATEGORIES["conditions"]:
            if f" {condition} " in f" {text} " or text.startswith(f"{condition} ") or text.endswith(f" {condition}") or text == condition:
                return condition
        return None
    
    def analyze_frequent_terms(self) -> Dict[str, int]:
        """Analyze frequent terms in query history for potential knowledge base expansion"""
        if not self.query_history:
            return {}
            
        # Combine all queries
        all_text = " ".join(self.query_history)
        
        # Normalize
        normalized = self._normalize_text(all_text)
        
        # Split into words
        words = normalized.split()
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Filter out common words
        common_words = set(["the", "a", "an", "in", "on", "at", "and", "or", "this", "that", "what", "why", "how", "is", "are"])
        filtered_counts = {word: count for word, count in word_counts.items() 
                          if word not in common_words and len(word) > 2 and count > 1}
        
        return dict(sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True))
    
    def update_knowledge_base(self, new_terms: Dict[str, List[Tuple[str, str]]]) -> bool:
        """
        Update knowledge base with new terms
        
        Args:
            new_terms: Dictionary with category keys and list of (term, definition) tuples
            
        Returns:
            Success status
        """
        try:
            # Load current knowledge base
            current_kb = self._load_knowledge_base()
            
            # Add new terms to appropriate categories
            for category, terms in new_terms.items():
                if category not in current_kb:
                    current_kb[category] = {}
                
                for term, definition in terms:
                    current_kb[category][term] = definition
            
            # Save updated knowledge base
            with open(self.KNOWLEDGE_BASE_FILE, 'w') as f:
                json.dump(current_kb, f, indent=2)
                
            # Refresh local knowledge base
            self.knowledge_base = current_kb
            
            return True
        except Exception as e:
            logger.error(f"Error updating knowledge base: {str(e)}")
            return False