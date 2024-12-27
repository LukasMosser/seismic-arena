from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from fastapi import Request, Response
import modal

from score_db import Model as ModelEnum
from models.unet_model import DeepFaultBaselineModel

volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)
MODEL_DIR = Path("/models")

app_image = modal.Image.debian_slim(python_version="3.12").pip_install(
     "huggingface_hub[hf_transfer]", "matplotlib", "safetensors", "pydantic", "fastapi==0.115.6", "torchmetrics", "torch", "torchvision", "albumentations", "Pillow", "huggingface_hub", "lightning", "datasets"
     ).env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})

app = modal.App(
    image=app_image,
    name="seisbase-endpoints",
    secrets=[modal.Secret.from_name("huggingface-secret-seisbase")],
    volumes={MODEL_DIR: volume},
    
)

@app.cls(timeout=3600, gpu="l40s", volumes={MODEL_DIR: volume}, secrets=[modal.Secret.from_name("huggingface-secret-seisbase")])
class Model:

    @modal.build()
    def build_models(self):
        for model in ModelEnum:
            DeepFaultBaselineModel.from_pretrained(model.value, cache_dir=MODEL_DIR)

    @modal.enter()
    def initialize_model(self):
        self.transformation = Compose(
            [
                Resize(896, 896),
                Normalize(mean=(0.5), std=(1.0)),
                ToTensorV2()
            ]
        )

    @modal.method()
    def generate_fault_likelihood(self, model, seismic_array: np.ndarray):
        import torch
        from PIL import Image
        self.seismic_predictor = DeepFaultBaselineModel.from_pretrained(model, cache_dir=MODEL_DIR, local_files_only=True)
        self.seismic_predictor.model = self.seismic_predictor.model.float().cuda()  # Convert model weights to float32
        
        # transform the image
        seismic_image = self.transformation(image=seismic_array)["image"]

        seismic_image = seismic_image.unsqueeze(0).float().cuda()
        
        # run the model on GPU
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float32):
            self.seismic_predictor.model.eval()
            predicted_prob = self.seismic_predictor.model(seismic_image.cuda()).softmax(dim=1)

        # convert the output to float32
        predicted_prob = predicted_prob.float()
        # convert the output to an image
        predicted_image = predicted_prob[:, 1].squeeze().cpu().numpy().astype(np.float32)
        predicted_image = Image.fromarray(predicted_image)

        return predicted_image

@app.function()
@modal.web_endpoint(method="POST", docs=True)
async def predict(request: Request):
        import base64
        import json
        # Takes a numpy array of floats as input, returns a bounding box image as a datauri
        body = await request.json()
        seismic_array = np.array(body["img"], dtype=np.float32)
        model_name = body["model"]
        img_data_out = Model().generate_fault_likelihood.remote(model_name, seismic_array)
        
        # Convert the image to bytes
        buffer = BytesIO()
        img_data_out.save(buffer, format="TIFF")
        img_data_out_bytes = buffer.getvalue()
        
        output_data = b"data:image/tiff;base64," + base64.b64encode(img_data_out_bytes)
        return Response(content=output_data)
