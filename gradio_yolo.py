import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLO

model = YOLO(r'C:\Users\artur\PycharmProjects\CV_Solar_Panels\model.pt')


def predict(image):
    try:
        image = np.array(image)

        results = model(image)
        result_image = results[0].plot()

        return Image.fromarray(result_image)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error"


iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil")
)

iface.launch()
