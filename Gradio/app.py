import os
import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load the model
model = YOLO('best.pt')

# Path to the photos folder
photos_folder = "Photos"

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path)
            images.append((img, filename))
    return images

def predict(image):
    try:
        image = np.array(image)
        results = model(image)
        result_image = results[0].plot()
        return Image.fromarray(result_image)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error"

def load_image_from_gallery(images, index):
    if images and 0 <= index < len(images):
        image = images[index]
        if isinstance(image, tuple):
            image = image[0]
        return image
    return None

def gallery_click_event(images, evt: gr.SelectData):
    index = evt.index
    selected_img = load_image_from_gallery(images, index)
    return selected_img

def clear_image():
    return None

# Load images at the start
images = load_images_from_folder(photos_folder)

with gr.Blocks(css=".container { background-color: white; }") as demo:
    with gr.Row():
        with gr.Column():
            selected_image = gr.Image(label="Selected Image from Gallery", type="pil")
            clear_button = gr.Button("Clear")
        
        with gr.Column():
            image_gallery = gr.Gallery(label="Image Gallery", elem_id="gallery", type="pil", value=[img for img, _ in images])
        
        with gr.Column():
            result_image = gr.Image(label="Result Image", type="pil")

    image_gallery.select(
        fn=gallery_click_event, 
        inputs=image_gallery, 
        outputs=selected_image
    )
    
    selected_image.change(
        fn=predict, 
        inputs=selected_image, 
        outputs=result_image
    )

    clear_button.click(
        fn=clear_image,
        inputs=None,
        outputs=selected_image
    )

demo.launch()
