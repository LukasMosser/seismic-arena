import csv
import io
import json
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List

import pandas as pd
from fastapi import Response
from modal import web_endpoint
import modal
from pydantic import BaseModel

from rating import compute_mle_elo

# -----------------------
# Data Model Definition
# -----------------------
class ExperienceEnum(int, Enum):
    novice = 1
    intermediate = 2
    expert = 3

class Winner(str, Enum):
    model_a = "model_a"
    model_b = "model_b"
    tie = "tie"


class Model(str, Enum):
    porestar_deepfault_unet_baseline_no_augment = "porestar/deepfault-unet-baseline-no-augment"
    porestar_deepfault_unet_baseline_weak_augment = "porestar/deepfault-unet-baseline-weak-augment"
    porestar_deepfault_unet_baseline_strong_augment = "porestar/deepfault-unet-baseline-strong-augment"

class Battle(BaseModel):
    model_a: Model
    model_b: Model
    winner: Winner
    judge: str
    image_idx: int 
    experience: ExperienceEnum = ExperienceEnum.novice
    tstamp: str = str(datetime.now())

class EloRating(BaseModel):
    model: Model
    elo_rating: float

# -----------------------
# Modal Configuration
# -----------------------

# Create a volume to persist data
data_volume = modal.Volume.from_name("seisbase-data", create_if_missing=True)

JSON_FILE_PATH = Path("/data/battles.json")
RESULTS_FILE_PATH = Path("/data/ratings.csv")

app_image = modal.Image.debian_slim(python_version="3.10").pip_install("pandas", "scikit-learn", "tqdm", "sympy")

app = modal.App(
    image=app_image,
    name="seisbase-eval",
    volumes={"/data": data_volume},
)

def ensure_json_file():
    """Ensure the JSON file exists and is initialized with an empty array if necessary."""
    if not os.path.exists(JSON_FILE_PATH):
        JSON_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(JSON_FILE_PATH, "w") as f:
            json.dump([], f)

def append_to_json_file(data):
    """Append data to the JSON file."""
    ensure_json_file()
    try:
        with open(JSON_FILE_PATH, "r+") as f:
            try:
                battles = json.load(f)
            except json.JSONDecodeError:
                # Reset the file if corrupted
                battles = []
            battles.append(data)
            f.seek(0)
            json.dump(battles, f, indent=4)
            f.truncate()
    except Exception as e:
        raise RuntimeError(f"Failed to append data to JSON file: {e}")

def read_json_file():
    """Read data from the JSON file."""
    ensure_json_file()
    try:
        with open(JSON_FILE_PATH, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []  # Return an empty list if the file is corrupted
    except Exception as e:
        raise RuntimeError(f"Failed to read JSON file: {e}")

@app.function()
@web_endpoint(method="POST", docs=True)
def add_battle(battle: Battle):
    """Add a new battle to the JSON file."""
    append_to_json_file(battle.dict())
    return {"status": "success", "battle": battle.dict()}


@app.function()
@web_endpoint(method="GET", docs=True)
def export_csv():
    """Fetch all battles and return as CSV."""
    battles = read_json_file()

    # Create CSV in memory
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["model_a", "model_b", "winner", "judge", "imaged_idx", "experience", "tstamp"])
    writer.writeheader()
    writer.writerows(battles)

    csv_data = output.getvalue()
    return Response(content=csv_data, media_type="text/csv")

@app.function()
@web_endpoint(method="GET", docs=True)
def compute_ratings() -> List[EloRating]:
    """Compute ratings from battles."""
    battles = pd.read_json(JSON_FILE_PATH, dtype=[str, str, str, str, int, int, str]).sort_values(ascending=True, by=["tstamp"]).reset_index(drop=True)
    elo_mle_ratings = compute_mle_elo(battles)
    elo_mle_ratings.to_csv(RESULTS_FILE_PATH)
    
    df = pd.read_csv(RESULTS_FILE_PATH)
    df.columns = ["Model", "Elo rating"]
    df = df.sort_values("Elo rating", ascending=False).reset_index(drop=True)
    scores = []
    for i in range(len(df)):
        scores.append(EloRating(model=df["Model"][i], elo_rating=df["Elo rating"][i]))
    return scores

@app.local_entrypoint()
def main():
    print("Local entrypoint running. Check endpoints for functionality.")
