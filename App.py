import gradio as gr
import torch
import clip
import numpy as np
import random
import os
from PIL import Image
from ultralytics import YOLO
from gtts import gTTS
import uuid
import time
import tempfile

# ---- Model loading ----
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# YOLO models
yolo_model = YOLO('yolov8n.pt').to(device)
fashion_model = YOLO('best.pt').to(device)  # custom model if needed

# ---- Style prompts ----
style_prompts = {
    'drippy': [
        "avant-garde streetwear",
        "high-fashion designer outfit",
        "trendsetting urban attire",
        "luxury sneakers and chic accessories",
        "cutting-edge, bold style"
    ],
    'mid': [
        "casual everyday outfit",
        "modern minimalistic attire",
        "comfortable yet stylish look",
        "simple, relaxed streetwear",
        "balanced, practical fashion"
    ],
    'not_drippy': [
        "disheveled outfit",
        "poorly coordinated fashion",
        "unfashionable, outdated attire",
        "tacky, mismatched ensemble",
        "sloppy, uninspired look"
    ]
}

# ---- Clothing prompts + responses ----
clothing_prompts = [
    "t-shirt", "dress shirt", "blouse", "hoodie", "jacket", "sweater", "coat",
    "dress", "skirt", "pants", "jeans", "trousers", "shorts",
    "sneakers", "boots", "heels", "sandals",
    "cap", "hat", "scarf", "gloves", "bag", "accessory", "tank-top", "haircut"
]

response_templates = {
    'drippy': [
        "You're Drippy, bruh – fire {item}!",
        "{item} goes crazy, on god!",
        "Certified drippy with that {item}."
    ],
    'mid': [
        "Drop the {item} and you might get a text back.",
        "It's alright, but I'd upgrade the {item}.",
        "Mid fit alert. That {item} is holding you back."
    ],
    'not_drippy': [
        "Bro thought that {item} was tuff!",
        "Oh hell nah! Burn that {item}!",
        "Crimes against fashion, especially that {item}! Also… maybe get a haircut.",
        "Never walk out the house again with that {item}."
    ]
}

# For labeling the category in the final output
CATEGORY_LABEL_MAP = {
    "drippy": "drippy",
    "mid": "mid",
    "not_drippy": "trash"  # rename "not_drippy" => "trash" in the final output
}

# Combine all prompts for CLIP processing
all_prompts = []
for cat_prompts in style_prompts.values():
    all_prompts.extend(cat_prompts)
all_prompts.extend(clothing_prompts)

def get_top_clothing(probs, n=3):
    """Retrieve top clothing items from CLIP probabilities."""
    clothing_probs = probs[len(all_prompts) - len(clothing_prompts):]
    top_indices = np.argsort(clothing_probs)[-n:]
    return [clothing_prompts[i] for i in reversed(top_indices)]

# ---- The main function to analyze an uploaded image ----
def analyze_outfit(img: Image.Image):
    # 1) YOLO detection to find the person region:
    results = yolo_model(img)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()

    # Find person bounding box
    person_indices = np.where(classes == 0)[0]
    cropped_img = img
    if len(person_indices) > 0:
        max_conf_idx = np.argmax(confidences[person_indices])
        x1, y1, x2, y2 = map(int, boxes[person_indices][max_conf_idx])
        cropped_img = img.crop((x1, y1, x2, y2))

    # 2) CLIP analysis
    image_tensor = clip_preprocess(cropped_img).unsqueeze(0).to(device)
    text_tokens = clip.tokenize(all_prompts).to(device)
    with torch.no_grad():
        logits, _ = clip_model(image_tensor, text_tokens)
        probs = logits.softmax(dim=-1).cpu().numpy()[0]

    # Style classification
    drip_len = len(style_prompts['drippy'])
    mid_len = len(style_prompts['mid'])
    not_len = len(style_prompts['not_drippy'])

    drip_score = np.mean(probs[:drip_len])
    mid_score = np.mean(probs[drip_len : drip_len + mid_len])
    not_score = np.mean(probs[drip_len + mid_len : drip_len + mid_len + not_len])

    if drip_score > mid_score and drip_score > not_score:
        category_key = 'drippy'
    elif mid_score > not_score:
        category_key = 'mid'
    else:
        category_key = 'not_drippy'

    # Map the internal key to the label we want to show the user
    category_label = CATEGORY_LABEL_MAP[category_key]

    # Clothing items
    clothing_items = get_top_clothing(probs)
    clothing_item = clothing_items[0]

    # response
    response = random.choice(response_templates[category_key]).format(item=clothing_item)

    # 3) Generate TTS with gTTS and save MP3
    tts_path = os.path.join(tempfile.gettempdir(), f"drip_{uuid.uuid4().hex}.mp3")
    tts = gTTS(response, lang="en")
    tts.save(tts_path)

    # Return the category, mp3, and text response
    return category_label, tts_path, response

# ---- Build the Gradio interface ----
demo = gr.Interface(
    fn=analyze_outfit,
    inputs=gr.Image(type='pil'),
    # We have THREE outputs now: text, audio, text
    outputs=[
        gr.Textbox(label="Category"),
        gr.Audio(label="Audio Response"),
        gr.Textbox(label="Response")
    ],
    title="Drip Detective 3000",
    description="Upload an image of your outfit to see if it's Drippy, Mid, or 'Trash'."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
