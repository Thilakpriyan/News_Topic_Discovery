from fastapi import FastAPI
from pydantic import BaseModel

# import your existing topic discovery function
# example: from model import get_topics

api = FastAPI()

class TextInput(BaseModel):
    text: str

@api.post("/predict")
def predict_topic(data: TextInput):
    text = data.text

    # CALL YOUR EXISTING LOGIC HERE
    # Example output (replace with your real result)
    topics = ["Infrastructure"]
    keywords = ["wifi", "hostel", "rooms"]

    return {
        "topics": topics,
        "keywords": keywords
    }
