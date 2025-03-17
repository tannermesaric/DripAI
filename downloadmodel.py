import cv2
import time
import torch
import clip
import os
from gtts import gTTS
from pygame import mixer
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO
import random
from threading import Thread
import tempfile
import uuid
from threading import Lock

# Add at the top with other imports
mixer.init()

# Audio configuration
audio_lock = Lock()
AUDIO_TEMP_DIR = os.path.join(tempfile.gettempdir(), "drip_audio_temp")
os.makedirs(AUDIO_TEMP_DIR, exist_ok=True)

class DripDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drip Detective 3000")
        
        # Load ML models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.yolo_model = YOLO('yolov8n.pt').to(self.device)
        self.fashion_model = YOLO('yolov8x-cls.pt').to(self.device)
        
        # GUI Setup
        self.create_widgets()
        self.setup_webcam()
        
        # Analysis variables
        self.capture_time = None
        self.processing = False
        self.clothing_item = ""
        self.category = ""
        self.response = ""

    def create_widgets(self):
        # Video display
        self.video_label = tk.Label(self.root)
        self.video_label.pack(padx=10, pady=10)
        
        # Control buttons
        self.btn_frame = tk.Frame(self.root)
        self.btn_frame.pack(pady=10)
        
        self.analyze_btn = tk.Button(
            self.btn_frame, 
            text="Analyze Outfit (Q)", 
            command=self.start_analysis,
            state=tk.NORMAL
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        self.quit_btn = tk.Button(
            self.btn_frame, 
            text="Exit", 
            command=self.close_app
        )
        self.quit_btn.pack(side=tk.LEFT, padx=5)
        
        # Results display
        self.results_frame = tk.Frame(self.root)
        self.results_frame.pack(pady=10)
        
        self.category_label = tk.Label(
            self.results_frame, 
            text="Category: ", 
            font=('Helvetica', 14, 'bold')
        )
        self.category_label.pack(anchor=tk.W)
        
        self.item_label = tk.Label(
            self.results_frame, 
            text="Detected Item: ", 
            font=('Helvetica', 12)
        )
        self.item_label.pack(anchor=tk.W)
        
        self.feedback_label = tk.Label(
            self.results_frame, 
            text="Feedback: ", 
            font=('Helvetica', 12, 'italic'),
            wraplength=400
        )
        self.feedback_label.pack(anchor=tk.W)

    def setup_webcam(self):
        self.cap = cv2.VideoCapture(0)
        self.show_video()
        
    def show_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.root.after(10, self.show_video)

    def start_analysis(self):
        if not self.processing:
            self.processing = True
            self.analyze_btn.config(state=tk.DISABLED)
            self.countdown(5)
            # Schedule analysis to start after 5 seconds
            self.root.after(5000, self.start_analysis_thread)

    def start_analysis_thread(self):
        Thread(target=self.run_analysis).start()

    def countdown(self, remaining):
        if remaining > 0:
            self.analyze_btn.config(text=f"Analyzing in {remaining}s...")
            self.root.after(1000, self.countdown, remaining-1)
        else:
            self.analyze_btn.config(text="Capturing image...")

    def run_analysis(self):
        # Capture image after delay
        ret, frame = self.cap.read()
        if ret:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cropped_image = self.analyze_region(pil_image)
            
            # CLIP processing
            image = self.preprocess(cropped_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                text = clip.tokenize(all_prompts).to(self.device)
                logits, _ = self.clip_model(image, text)
                probs = logits.softmax(dim=-1).cpu().numpy()[0]

            # Calculate scores and determine category
            drip_score = np.mean(probs[:len(style_prompts['drippy'])])
            mid_score = np.mean(probs[len(style_prompts['drippy']): 
                                    len(style_prompts['drippy'])+len(style_prompts['mid'])])
            not_score = np.mean(probs[len(style_prompts['drippy'])+len(style_prompts['mid']): 
                                    len(style_prompts['drippy'])+len(style_prompts['mid'])+len(style_prompts['not_drippy'])])
            
            self.category = 'drippy' if drip_score > mid_score and drip_score > not_score else \
                        'mid' if mid_score > not_score else 'not_drippy'
            
            # Get clothing items
            clothing_items = get_top_clothing(probs)
            self.clothing_item = ", ".join(clothing_items)
        
        # Generate response
        self.response = random.choice(response_templates[self.category]).format(item=clothing_items[0])
        
        # Update GUI in main thread
        self.root.after(0, self.update_results)
        
        # Text-to-speech
        self.play_tts()
    
        # Reset processing state in main thread
        self.root.after(0, lambda: self.analyze_btn.config(state=tk.NORMAL, text="Analyze Outfit (Q)"))
        self.processing = False

    def countdown(self, remaining):
        if remaining > 0:
            self.analyze_btn.config(text=f"Analyzing in {remaining}s...")
            self.root.after(1000, self.countdown, remaining-1)
        else:
            self.analyze_btn.config(text="Analyze Outfit (Q)")

    def update_results(self):
        self.category_label.config(text=f"Category: {self.category.upper()}")
        self.item_label.config(text=f"Detected Item: {self.clothing_item}")
        self.feedback_label.config(text=f"Feedback: {self.response}")

    def play_tts(self):
        try:
            with audio_lock:
                temp_file = os.path.join(AUDIO_TEMP_DIR, f"drip_{uuid.uuid4().hex}.mp3")
                
                tts = gTTS(self.response, lang="en")
                tts.save(temp_file)
                
                mixer.music.load(temp_file)
                mixer.music.play()
                
                while mixer.music.get_busy():
                    time.sleep(0.1)
                
                mixer.music.unload()
                self.safe_delete(temp_file)
                
        except Exception as e:
            print(f"Audio error: {e}")

    def safe_delete(self, path):
        for _ in range(3):
            try:
                if os.path.exists(path):
                    os.remove(path)
                    return
                time.sleep(0.1)
            except Exception as e:
                print(f"Delete failed: {e}")
                time.sleep(0.2)

    def close_app(self):
        # Clean audio files
        for filename in os.listdir(AUDIO_TEMP_DIR):
            file_path = os.path.join(AUDIO_TEMP_DIR, filename)
            self.safe_delete(file_path)
            
        self.cap.release()
        self.root.destroy()

    def analyze_region(self, pil_image):
        results = self.yolo_model(pil_image)
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        
        person_indices = np.where(classes == 0)[0]
        if len(person_indices) > 0:
            max_conf_idx = np.argmax(confidences[person_indices])
            x1, y1, x2, y2 = map(int, boxes[person_indices][max_conf_idx])
            return pil_image.crop((x1, y1, x2, y2))
        return pil_image

    def close_app(self):
        self.cap.release()
        self.root.destroy()

# Style prompts and other constants (keep your existing definitions)
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

clothing_prompts = [
    "t-shirt", "dress shirt", "blouse", "hoodie", "jacket", "sweater", "coat",
    "dress", "skirt", "pants", "jeans", "trousers", "shorts",
    "sneakers", "boots", "heels", "sandals",
    "cap", "hat", "scarf", "gloves", "bag", "accessory", "tank-top", "haircut"
]

# Combine all prompts for CLIP processing
all_prompts = []
for category in style_prompts.values():
    all_prompts.extend(category)
all_prompts.extend(clothing_prompts)

# Response templates with clothing-specific feedback
response_templates = {
    'drippy': [
        "Your Drippy bruh, fire {item}",
        "{item} goes crazy bruh on god",
        "Certified drippyfile with that {item}"
    ],
    'mid': [
        "Drop the {item} and you could get a text back",
        "It's aight, but upgrade the {item}",
        "Middddd fit alert. The {item} is holding you back"
    ],
    'not_drippy': [
        "Bro thought that {item} was tuff!",
        "Aw hell nah! Burn that {item}!",
        "Crimes against fashion! Especially the {item}, also...get a haircut",
        "yeah bro just never walk out the house again with the {item}"
        "Maybe it isnt the {item}...you may just need a haircut"
    ]
}


def get_top_clothing(probs, n=3):
    clothing_probs = probs[len(all_prompts)-len(clothing_prompts):]
    top_indices = np.argsort(clothing_probs)[-n:]
    return [clothing_prompts[i] for i in reversed(top_indices)]

if __name__ == "__main__":
    root = tk.Tk()
    app = DripDetectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close_app)
    root.mainloop()