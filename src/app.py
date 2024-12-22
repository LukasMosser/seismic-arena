import base64
import io
import random
from io import BytesIO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
from datasets import load_dataset
import gradio as gr

from score_db import Battle
from score_db import Model as ModelEnum, Winner

def make_plot(seismic, predicted_image):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(Image.fromarray(seismic), cmap="gray")
    ax.imshow(predicted_image, cmap="Reds", alpha=0.5, vmin=0, vmax=1)
    ax.set_axis_off()
    fig.canvas.draw()

    # Create a bytes buffer to save the plot
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    # Open the PNG image from the buffer and convert it to a NumPy array
    image = np.array(Image.open(buf))
    return image

def call_endpoint(model: ModelEnum, img_array, url: str="https://lukasmosser--seisbase-endpoints-predict.modal.run"):
    response = requests.post(url, json={"img": img_array.tolist(), "model": model})

    if response:
        # Parse the base64-encoded image data
        if response.text.startswith("data:image/tiff;base64,"):
            img_data_out = base64.b64decode(response.text.split(",")[1])
            predicted_image = np.array(Image.open(BytesIO(img_data_out)))
            return predicted_image

def select_random_image(dataset):
    idx = random.randint(0, len(dataset))
    return idx, np.array(dataset[idx]["seismic"])

def select_random_models():
    model_a = random.choice(list(ModelEnum))
    model_b = random.choice(list(ModelEnum))
    return model_a, model_b


# Create a Gradio interface
with gr.Blocks() as evaluation:
    gr.Markdown("""
    ## Seismic Fault Detection Model Evaluation
    This application allows you to compare the performance of different seismic fault detection models. 
    Two models are selected randomly, and their predictions are displayed side by side. 
    You can choose the better model or mark it as a tie. The results are recorded and used to update the model ratings.
    """)

    battle = gr.State([])
    radio = gr.Radio(choices=["Less than 5 years", "5 to 20 years", "more than 20 years"], label="How much experience do you have in seismic fault interpretation?")
    with gr.Row():
        output_img1 = gr.Image(label="Model A Image")
        output_img2 = gr.Image(label="Model B Image")

    def show_images():
        dataset = load_dataset("porestar/crossdomainfoundationmodeladaption-deepfault", split="valid")
        idx, image_1 = select_random_image(dataset)
        model_a, model_b = select_random_models()
        fault_probability_1 = call_endpoint(model_a, image_1)
        fault_probability_2 = call_endpoint(model_b, image_1)

        img_1 = make_plot(image_1, fault_probability_1)
        img_2 = make_plot(image_1, fault_probability_2)
        experience = 1 
        if radio.value == "5 to 20 years":
            experience = 2
        elif radio.value == "more than 20 years":
            experience = 3
        battle.value.append(Battle(model_a=model_a, model_b=model_b, winner="tie", judge="None", experience=experience, image_idx=idx))
        return img_1, img_2
    
    # Define the function to make an API call
    def make_api_call(choice: Winner):
        api_url = "https://lukasmosser--seisbase-eval-add-battle.modal.run"
        battle_out = battle.value 
        battle_out[-1].winner = choice
        experience = 1 
        if radio.value == "5 to 20 years":
            experience = 2
        elif radio.value == "more than 20 years":
            experience = 3
        battle_out[-1].experience = experience
        response = requests.post(api_url, json=battle_out[-1].dict())

    # Load images on startup
    evaluation.load(show_images, inputs=[], outputs=[output_img1, output_img2])
    
    with gr.Row():
        btn_winner_a = gr.Button("Winner Model A")
        btn_tie = gr.Button("Tie")
        btn_winner_b = gr.Button("Winner Model B")

    # Define button click events
    btn_winner_a.click(lambda: make_api_call(Winner.model_a), inputs=[], outputs=[]).then(show_images, inputs=[], outputs=[output_img1, output_img2])
    btn_tie.click(lambda: make_api_call(Winner.tie), inputs=[], outputs=[]).then(show_images, inputs=[], outputs=[output_img1, output_img2])
    btn_winner_b.click(lambda: make_api_call(Winner.model_b), inputs=[], outputs=[]).then(show_images, inputs=[], outputs=[output_img1, output_img2])

with gr.Blocks() as leaderboard:
    def get_results():
        response = requests.get("https://lukasmosser--seisbase-eval-compute-ratings.modal.run")
        data = response.json()

        models = [entry["model"] for entry in data]
        elo_ratings = [entry["elo_rating"] for entry in data]

        fig, ax = plt.subplots()
        ax.barh(models, elo_ratings, color='skyblue')
        ax.set_xlabel('ELO Rating')
        ax.set_title('Model ELO Ratings')
        plt.tight_layout()

        fig.canvas.draw()

        # Create a bytes buffer to save the plot
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        # Open the PNG image from the buffer and convert it to a NumPy array
        image = np.array(Image.open(buf))
        return image
    
    with gr.Row():
        elo_ratings = gr.Image(label="ELO Ratings")
    
    leaderboard.load(get_results, inputs=[], outputs=[elo_ratings])

demo = gr.TabbedInterface([evaluation, leaderboard], ["Arena", "Leaderboard"])

# Launch the interface
if __name__ == "__main__":
    demo.launch(show_error=True)

