import spacy
from spacy.matcher import PhraseMatcher
import re

# Load models
nlp = spacy.load("en_ner_bc5cdr_md")

# Create medical terminology matchers
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

# Add anatomical terms
anatomy_terms = ["abdomen", "chest", "brain", "lung", "heart", "liver", "kidney", "spine", "joint", "bone", "tissue"]
anatomy_patterns = [nlp(term) for term in anatomy_terms]
matcher.add("ANATOMY", None, *anatomy_patterns)

# Add modality terms
modality_terms = ["x-ray", "mri", "ct scan", "ultrasound", "pet scan", "radiograph", "angiogram", "mammogram"]
modality_patterns = [nlp(term) for term in modality_terms]
matcher.add("MODALITY", None, *modality_patterns)

def extract_medical_entities(text):
    """Extract medical entities from text using NER and pattern matching."""
    doc = nlp(text)
    
    # Get NER entities
    ner_entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    
    # Get pattern matches
    matches = matcher(doc)
    pattern_entities = []
    for match_id, start, end in matches:
        match_text = doc[start:end].text
        match_label = nlp.vocab.strings[match_id]
        pattern_entities.append({"text": match_text, "label": match_label})
    
    # Combine entities
    all_entities = ner_entities + pattern_entities
    
    # Add uncertainty detection
    uncertainty_terms = ["possible", "probable", "maybe", "might be", "could be", "suspect", "suspicious for"]
    has_uncertainty = any(term in text.lower() for term in uncertainty_terms)
    
    return {
        "entities": all_entities,
        "has_uncertainty": has_uncertainty,
        "entity_count": len(all_entities)
    }

def extract_medical_relationships(text):
    """Extract relationships between medical entities."""
    doc = nlp(text)
    relationships = []
    
    # Look for basic relationships between entities
    for ent1 in doc.ents:
        for ent2 in doc.ents:
            if ent1 != ent2:
                # Check if entities are close to each other (within 5 tokens)
                if abs(ent1.start - ent2.start) < 5:
                    relationships.append({
                        "entity1": ent1.text,
                        "entity1_type": ent1.label_,
                        "entity2": ent2.text,
                        "entity2_type": ent2.label_,
                        "relation_type": "CO_OCCURRENCE"
                    })
    
    return relationships