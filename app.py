from typing import Union
from fastapi import FastAPI
from transformer import PhiTransformer

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to Llama-Engine!"}


@app.get("/download")
def download(model_id: Union[str, None] = None):
    if not model_id:
        model_id = None
    phi_transformer = PhiTransformer(model_id=model_id)
    return phi_transformer.download_phi_3()
