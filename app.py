import gradio as gr
import requests
import os

API_URL = "https://api-inference.huggingface.co/models/stabilityai/TripoSR"
HEADERS = {"Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"}

def generate_3d_from_image(image):
    with open(image, "rb") as f:
        response = requests.post(API_URL, headers=HEADERS, files={"file": f})

    if response.status_code == 200:
        output_path = "output.glb"
        with open(output_path, "wb") as out_file:
            out_file.write(response.content)
        return output_path
    else:
        return f"Error: {response.status_code} - {response.text}"

with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Genera un modelo 3D real desde una imagen usando TripoSR")
    with gr.Row():
        image_input = gr.Image(type="filepath", label="Sube una imagen (.png o .jpg)")
    generate_btn = gr.Button("Generar modelo 3D")
    output_model = gr.Model3D(label="Vista previa del modelo 3D")

    generate_btn.click(fn=generate_3d_from_image, inputs=image_input, outputs=output_model)

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
