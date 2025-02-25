import os
import pandas as pd
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_experimental.agents import create_csv_agent
from langchain_groq import ChatGroq
from fastapi.middleware.cors import CORSMiddleware

#  Setup Logging for Better Debugging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#  Load Groq API Key (Ensure this is kept secure in production)
os.environ["GROQ_API_KEY"] = "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"

# Initialize FastAPI Application
app = FastAPI()

# Enable CORS to allow frontend communication (Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change this for better security)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Load Titanic Dataset (only once to improve performance)
DATASET_PATH = "tested.csv"

try:
    df = pd.read_csv(DATASET_PATH)
    logging.info(" Titanic dataset loaded successfully.")
except Exception as e:
    logging.error(f" Error loading dataset: {e}")
    raise RuntimeError("Dataset loading failed. Please check the file path.")

#  Initialize LLM (Groq's Llama3-8b-8192)
llm = ChatGroq(model_name="llama3-8b-8192", temperature=0)

#  Create CSV Agent for querying Titanic dataset
agent = create_csv_agent(llm, DATASET_PATH, verbose=True, allow_dangerous_code=True)

#  Define API Request Model
class ChatRequest(BaseModel):
    message: str

# Root Route
@app.get("/")
async def home():
    return {"message": "Titanic Chatbot API is running!"}

#  Chatbot API Endpoint
@app.post("/chat/")
async def chat(input: ChatRequest):
    query = input.message.strip().lower()
    logging.info(f" Received user query: {query}")

    try:
        # Step 1: Check if the query can be answered using the Titanic dataset
        response = agent.invoke(query)

        # Extract only the relevant response
        clean_response = response.get("output", response) if isinstance(response, dict) else response

        if clean_response:
            logging.info(f" Responding with dataset-based answer.")
            return {"response": clean_response}

        #  Step 2: If dataset doesn't answer, fall back to Groq AI model
        logging.info(" Querying Groq AI model for response...")
        ai_response = llm.invoke(query)

        return {"response": ai_response}
    
    except Exception as e:
        logging.error(f" Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Something went wrong while processing your request.")

#  Start FastAPI Server (only runs when executing this script directly)
if __name__ == "__main__":
    import uvicorn
    logging.info("🚀 Starting FastAPI Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
