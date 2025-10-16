# automated-lender: An AI Lending Risk ReAct Agent
An intelligent **ReAct Agent chatbot** that processes lending requests and evaluates loan default risk using a deployed **XGBoost machine learning model**.  
This agent doesn’t just predict outcomes — it **explains its reasoning** through **SHAP-based interpretability** and **Retrieval-Augmented Generation (RAG)** capabilities.

[👉 Try it on Hugging Face](https://huggingface.co/spaces/aribarrios/automated-lender)

## Project Overview

The **AI Lending Risk ReAct Agent** combines natural language reasoning, machine learning, and explainable AI to guide users through the loan approval process.  

- **Conversational Interface:** The agent interacts with users to collect key lending features such as income, loan amount, term, and interest rate.  
- **Machine Learning Decisioning:** The collected inputs are passed to a trained **XGBoost** model that predicts whether a borrower is likely to **default or not**.  
- **Web-Enabled Intelligence:** The agent can **perform live web searches** to supplement its reasoning or clarify financial concepts during the interaction.  
- **Explainable AI with SHAP:** After making a prediction, the agent uses **SHAP values** to break down which features most influenced the outcome, summarizing these insights in plain language for the user.  
- **RAG for Deeper Insights:** A **Retrieval-Augmented Generation (RAG)** pipeline helps the agent interpret and describe key SHAP metrics more intelligently, grounding explanations in financial context.


## Architecture Overview

The system is built with modular AI and ML components that integrate through **LangChain** and **LangGraph**, and is deployed on **Hugging Face** for easy interaction.

## Tech Stack:
- Python  
- XGBoost  
- LangChain & LangGraph  
- SHAP (for interpretability)  
- Deployed on Hugging Face

## Key Features

- **ReAct-based Reasoning:** Combines reasoning and action for dynamic decision-making.  
- **Model Explainability:** Uses SHAP values to help users understand why a loan was approved or denied.  
- **Financial Context Awareness:** RAG pipeline enriches explanations with relevant financial knowledge.  
- **Interactive Deployment:** Hosted on Hugging Face for real-time user interaction. 