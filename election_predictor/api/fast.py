import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from election_predictor.interface.main.py import predict_election

# Instantiating FastAPI and
app = FastAPI()
app.state.model = predict_election()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Creating endpoint for predicting national vote share
@app.get("/predict_vote_share")
def predict_vote_share():
    prediction = predict_election()
    return prediction



@app.get("/")
def root():
    return {'introduction': 'Welcome to our API, please use our endpoints\
            to get national vote share or constituency results.'}
