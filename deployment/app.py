from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize the FastAPI app
app = FastAPI()

# Load the pre-trained model
model = joblib.load('logistic_regression_model.pkl')

# Define the request body structure
class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Define the prediction route
@app.post("/predict")
def predict(iris: IrisData):
    try:
        # Convert input data to DataFrame format
        input_data = pd.DataFrame([[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]],
                                  columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
        # Make a prediction
        prediction = model.predict(input_data)
        # Map the prediction to the corresponding class
        target_names = ['setosa', 'versicolor', 'virginica']
        return {"prediction": target_names[prediction[0]]}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in prediction: {str(e)}")

# Root endpoint to check if the app is running
@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classifier API!"}
