# SHL Assessment Recommendation Engine

## Overview
This project implements a semantic recommendation engine that suggests suitable SHL assessments based on a free-text hiring requirement. It helps recruiters quickly identify relevant assessments without manually browsing the catalogue.

## Approach
- SHL assessment metadata is stored as structured text
- Sentence embeddings are generated using a transformer-based embedding model
- FAISS is used for fast similarity search
- Given a hiring requirement in natural language, the system retrieves and ranks the most relevant assessments

## How It Works
1. Run `embed.py` to generate embeddings and build the FAISS index  
2. Run `retrieve.py` and enter a hiring requirement when prompted  
3. The top-matching SHL assessments are returned with brief details  

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
- OpenAI Embeddings  
- FAISS  
- NumPy  
