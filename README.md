# MediScan AI Assistant

MediScan AI is an NLP-powered medical image assistant that allows users to upload medical images (X-rays, MRIs, CT scans, etc.) and ask intelligent questions about them. By combining advanced image captioning (via BLIP-2) and medical named entity recognition (SciSpacy), the assistant provides clinically relevant insights, entity detection, and AI-driven reasoning—all through natural language queries.

---

## Project Highlights

- Image Understanding: Uses BLIP-2 for medical image captioning.
- Medical NLP: Extracts diseases, anatomy, and modality terms using `en_ner_bc5cdr_md`.
- Intent Detection: Classifies user questions (diagnosis, treatment, explanation, general).
- Query Enhancement: Rewrites vague user prompts into semantically rich medical instructions.
- LLM-Powered Reasoning: Connects to GROQ Cloud API (LLaMA-4 Scout) for multimodal answers.
- Interactive UI: Built with Tailwind CSS, featuring upload, chat, and NLP entity overlays.

---

## Architecture Overview

System flow:

1. User uploads image and query
2. Image analyzed by BLIP-2 for captioning
3. NLPProcessor extracts entities, detects intent, and enhances query
4. GROQ Cloud LLM processes enhanced query with image
5. AI response, detected entities, and intent displayed in UI

Refer to `assets/architecture.png` and `assets/dataflow.png` for full diagrams.

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourname/medical-image-analyzer.git
cd medical-image-analyzer
```

### 2. Create and activate virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install SciSpacy model manually

```bash
pip install https://huggingface.co/Kaelan/en_ner_bc5cdr_md/resolve/main/en_ner_bc5cdr_md-any-py3-none-any.whl
```

### 5. Set up `.env` file

Create a `.env` file with your GROQ API key:

```
GROQ_API_KEY=your_groq_api_key
```

---

## How to Use

### CLI Mode (Optional)

```bash
python main.py sample.jpg "What condition is this?" --type diagnosis
```

### Web App (Recommended)

```bash
uvicorn app:app --reload
```

- Navigate to: http://127.0.0.1:8000
- Upload an image and enter a question
- Receive LLM-generated response and detected medical terms

---

## NLP Stack

| Component         | Purpose                                              |
|------------------|------------------------------------------------------|
| SciSpacy Model   | Named entity recognition for diseases, anatomy, etc. |
| PhraseMatcher    | Pattern recognition for modality and anatomy terms   |
| Intent Detector  | Classifies queries into general, diagnosis, etc.     |
| Prompt Enhancer  | Reformulates user query using structured templates   |
| GROQ LLM         | Multimodal processing of enhanced query + image      |

---

## License

GNU GENERAL PUBLIC LICENSE
