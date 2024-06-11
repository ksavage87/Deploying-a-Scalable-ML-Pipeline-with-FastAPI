import os

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import uvicorn

class CensusData(BaseModel):
    age: int = Field(..., example=25)
    workclass: str = Field(..., example="Private")
    fnlwgt: int = Field(..., example=226802)
    education: str = Field(..., example="11th")
    education_num: int = Field(..., example=7, alias="education-num")
    marital_status: str = Field(..., example="Never-married", alias="marital-status")
    occupation: str = Field(..., example="Machine-op-inspct")
    relationship: str = Field(..., example="Own-child")
    race: str = Field(..., example="Black")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Census Prediction API"}

@app.post("/predict")
def predict(data: CensusData):
    # Dummy prediction logic
    prediction = "salary"
    return {"prediction": prediction}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    

# TODO: create a POST on a different path that does model inference
@app.post("/data/")
async def post_inference(data: Data):
    # DO NOT MODIFY: turn the Pydantic model into a dict.
    data_dict = data.dict()
    # DO NOT MODIFY: clean up the dict to turn it into a Pandas DataFrame.
    # Here it uses the functionality of FastAPI/Pydantic/etc to deal with this.
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data = pd.DataFrame.from_dict(data)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    
    data_processed, _, _, _ = process_data(
    data, 
    categorical_features=cat_features, 
    encoder=encoder, 
    training=False
)

    _inference = inference(model, data_processed)
    return {"result": apply_label(_inference)}
