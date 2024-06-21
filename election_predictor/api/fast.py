import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from election_predictor.interface.main import predict_election

# Instantiating FastAPI
app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Creating endpoint for predicting national vote share
# Returns a dictionary containing the predicted general
# election vote share and projected seats.
@app.get("/predict")
def predict():
    prediction = predict_election()
    return {"results": prediction}

# Define root endpoint
@app.get("/")
def root():
    return {'introduction': 'Welcome to our API, please use our endpoints to get national vote share or constituency results.'}
