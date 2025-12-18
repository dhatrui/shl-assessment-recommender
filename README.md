# SHL Assessment Recommendation Engine

## Overview
This project builds a semantic search–based recommendation system for SHL assessments using FAISS and sentence embeddings. 
Given a natural language job requirement or query, the system retrieves the most relevant SHL assessments and evaluates retrieval quality using Precision@K.

## Approach
- SHL assessment metadata is stored as structured text
- Sentence embeddings are generated using a transformer-based embedding model
- FAISS is used for fast similarity search
- Given a hiring requirement in natural language, the system retrieves and ranks the most relevant assessments

## How It Works
1. Install dependencies: pip install sentence-transformers faiss-cpu numpy
2. Run `embed.py` to generate embeddings and build the FAISS index  
3. Run `retrieve.py` and enter a hiring requirement when prompted  
4. The top-matching SHL assessments are returned with brief details  

## Repository Structure
data/
└── shl_catalogue.json
embed.py
retrieve.py
requirements.txt
.gitignore
README.md 


## Tech Stack
- Python  
- Sentence Transformers (all-MiniLM-L6-v2)  
- FAISS  
- NumPy  

## Note: For simplicity and reproducibility, the SHL catalogue is provided as a structured JSON file instead of live web scraping.
## Retrieval quality is evaluated using Precision@K on a small ground-truth set
