from fastapi import FastAPI, HTTPException
from pyprojroot import here
from pydantic import BaseModel, Field
import joblib
import onnxruntime as ort
import numpy as np
import torch

app = FastAPI(title="HPLC RT Prediction API", version="0.1.0")

# Some important paths
PROCESSED_DATA_PATH = here() / "data" / "processed"
SAVED_MODELS_PATH =  here() / "models"
SAVED_MULTILAYER_PERCEPTRON_MODEL_PATH = SAVED_MODELS_PATH / "multilayer_perceptron"

# Model names
model_names = [
    "mlp_fp_smart_retained",
    ]
#Let's load the data transformation pipelines
data_pipelines = {
    model_names[0]: joblib.load(SAVED_MULTILAYER_PERCEPTRON_MODEL_PATH / "fps_pipeline_for_rdkit_fingerprints_2026_07_03_v0.pkl")
}

models = {
    model_names[0]: ort.InferenceSession(
        SAVED_MULTILAYER_PERCEPTRON_MODEL_PATH / "multilayer_perceptron_rdkit_pipeline_2026_07_03_v0.onnx", 
        providers=["CPUExecutionProvider"]
    )
}

model_notes = {
    model_names[0]: "Fully connected NN architecture based on RDKit fingerprints with counts and a maxPath of 5. "
    "Model was trained on the SMRT dataset using only retained molecules."
}

class PredictionRequest(BaseModel):
    smiles: str = Field(description="Molecular SMILES of a single molecule")

class PredictionResponse(BaseModel):
    smiles: str = Field(description="Molecular SMILES of a single molecule")
    model_name: str =  Field(description="Name od the model used for the prediction")
    prediction: float = Field(description="Predicted retention time in seconds")
    model_notes: str = Field(description="relevant information about the model used")

@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "HPLC RT prediction API"}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/models_info")
def models_info() -> dict[str, str]:
    response = ""
    for model_name in model_names:
        response += f"{model_name}: {model_notes[model_name]}\n"

    return {"message": response}

@app.post("/predict/{model_name}")
def predict(model_name: str, request: PredictionRequest) -> PredictionResponse:
    if model_name not in model_names:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # 3. Select the model and perform inference
    data_pipeline = data_pipelines.get(model_name, None)
    model = models[model_name]
    processed_data = np.array([[request.smiles]])
    if data_pipeline is not None:
        processed_data = data_pipeline.transform(processed_data).astype(np.float32)
    #processed_data = torch.from_numpy(processed_data)
    input_name = model.get_inputs()[0].name
    prediction = model.run(None, {input_name: processed_data})
    print(prediction)
    print(type(prediction))
    print(len(prediction))
    
    return {"smiles": request.smiles, "model_name": model_name, "prediction": prediction[0][0][0], "model_notes": model_notes[model_name]}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("dl_hplc_smrt.main:app", host="0.0.0.0", port=8000, reload=True)
