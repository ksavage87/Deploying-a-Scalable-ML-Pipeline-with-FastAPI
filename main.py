from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
from train_model import process_data, inference, apply_label, load_model
import uvicorn
import pickle


encoder_path = (
    "/Users/kaleymayer/Deploying-a-Scalable-ML-Pipeline-with-FastAPI"
    "model/encoder.pkl"
)
model_path = (
    "/Users/kaleymayer/Deploying-a-Scalable-ML-Pipeline-with-FastAPI"
    "model/model.pkl"
)
encoder = load_model(encoder_path)
model = load_model(model_path)


class CensusData(BaseModel):
    age: int = Field(..., example=25)
    workclass: str = Field(..., example="Private")
    fnlwgt: int = Field(..., example=226802)
    education: str = Field(..., example="11th")
    education_num: int = Field(..., example=7, alias="education-num")
    marital_status: str = Field(
        ...,
        example="Never-married",
        alias="marital-status"
    )
    occupation: str = Field(..., example="Machine-op-inspct")
    relationship: str = Field(..., example="Own-child")
    race: str = Field(..., example="Black")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(
        ...,
        example="United-States",
        alias="native-country"
    )


class Data(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome to the Census Prediction API"}


@app.post("/predict")
def predict(data: CensusData):

    prediction = "salary"
    return {"prediction": prediction}


@app.post("/data/")
async def post_inference(data: Data):

    data_dict = data.dict()
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
