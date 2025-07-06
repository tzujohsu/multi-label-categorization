# Multi-Label News Categorization

A machine learning pipeline for  categorizing news articles into multiple categories using NLP and ML techniques.

## Live Demo

🌐 **Try it live**: [Multi-Label News Categorization Demo](https://jocelynhsutjh.streamlit.app/mlb-categorization)

## API Usage

Classify text via API:
```bash
curl "http://107.23.11.29/categorize?txt=[How is the weather today]"
```

Response format (Sample):
```json
{
  "category1": 0.85,
  "category2": 0.72,
  "category3": 0.45
}
```



## Overview

This project implements an end-to-end machine learning pipeline for multi-label news categorization. It uses advanced NLP techniques including text preprocessing, feature engineering, and machine learning models to classify news articles into 17 different categories.

## Features

- **Multi-label Classification**: Classifies articles into multiple categories simultaneously
- **NLP Pipeline**: Comprehensive text preprocessing including stopword removal and stemming
- **Multiple ML Models**: Support for LightGBM and Logistic Regression
- **Experiment Tracking**: MLflow integration for experiment management
- **API Deployment**: FastAPI-based REST API for model serving
- **Docker Support**: Containerized deployment

## Categories

The model can classify news articles into the following 17 categories:
- Crime, Law and Justice
- Arts, Culture, Entertainment and Media
- Economy, Business and Finance
- Disaster, Accident and Emergency Incident
- Environment
- Education
- Health
- Human Interest
- Lifestyle and Leisure
- Politics
- Labour
- Religion and Belief
- Science and Technology
- Society
- Sport
- Conflict, War and Peace
- Weather

## Project Structure
Only key files and folders are shown for clarity

```
├── src/                         
│   ├── ingest_data.py           
│   ├── filter_text.py          
│   ├── feature_engineering.py  
│   ├── data_splitter.py         
│   ├── model_building.py       
│   └── model_evaluator.py      
├── steps/                      
├── pipelines/             
├── data/                  
├── deployment/            
│   ├── app/              
│   ├── Dockerfile        
│   └── requirements.txt  
├── analysis/             
├── post_analysis/        
└── run_pipeline.py       # Pipeline execution script
```


### Model Training

The pipeline automatically:
1. Ingests news data from `data/archive.zip`
2. Preprocesses text (filtering, stopword removal, stemming)
3. Engineers features using TF-IDF vectorization
4. Splits data into training and test sets
5. Trains a LightGBM multi-label classifier (or any classifier you prefer)
6. Evaluates model performance

## About Model 

The current model uses:
- **Algorithm**: LightGBM with MultiOutputClassifier
- **Features**: TF-IDF vectorization
- **Text Preprocessing**: Stopword removal, stemming, text filtering
- **Evaluation**: Multi-label classification metrics

## Technologies Used

- **ML Framework**: Scikit-learn, LightGBM
- **NLP**: NLTK, TF-IDF vectorization
- **Pipeline**: ZenML
- **Experiment Tracking**: MLflow
- **API**: FastAPI
- **Deployment**: Docker, AWS
