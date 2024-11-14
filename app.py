# Import required libraries
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from gtts import gTTS
from IPython import display
import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Blip2ForConditionalGeneration, Blip2Processor
from peft import PeftModel, PeftConfig
import IndicTransToolkit 
from IndicTransToolkit.IndicTransToolkit import IndicProcessor

# Setup paths and models
PEFT_MODEL_ID = "Eye_blind_files/peft"
OFFLOAD_FOLDER = 'Eye_blind_files/offload'

# Setup device configurations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
device_map = {"query_tokens": 0, "vision_model": 0, "language_model": 1,
              "language_projection": 1, "lm_head": 1, "qformer": 0}
max_memory = {i: "4GB" for i in range(2)}  # 2 GPU

# Load BLIP2 model configuration
config = PeftConfig.from_pretrained(PEFT_MODEL_ID)
model_blip = Blip2ForConditionalGeneration.from_pretrained(
    config.base_model_name_or_path, device_map="auto", torch_dtype=torch.float16, offload_folder=OFFLOAD_FOLDER
)
model_blip = PeftModel.from_pretrained(model_blip, PEFT_MODEL_ID)
checkpoint = 'Salesforce/blip2-opt-2.7b'
processor = Blip2Processor.from_pretrained(checkpoint)

trans_dict = {"Bengali": ["ben_Beng","bn"],"English": ["eng_Latn","en"],"Gujarati": ["guj_Gujr","gu"],
    "Hindi": ["hin_Deva","hi"],"Kannada": ["kan_Knda","kn"],"Malayalam": ["mal_Mlym","ml"],"Marathi": ["mar_Deva","mr"],
    "Nepali": ["npi_Deva","ne"],"Punjabi": ["pan_Guru","pa"],"Tamil": ["tam_Taml","ta"],"Telugu": ["tel_Telu","te"],"Urdu": ["urd_Arab","ur"],}

example_images = [["Eye_blind_files/sample_images/3298199743_d8dd8f94a0.jpg"],["Eye_blind_files/sample_images/3298233193_d2a550840d.jpg"],["Eye_blind_files/sample_images/3299418821_21531b5b3c.jpg"],
                  ["Eye_blind_files/sample_images/3301754574_465af5bf6d.jpg"],["Eye_blind_files/sample_images/3302804312_0272091cd5.jpg"]]

#IndicTrans
model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
tokenizer_ind = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
ip = IndicProcessor(inference=True)

# Translate function for generating captions in different languages
def translate(eng_caption, tgt_lang):
    input_sentences = [eng_caption]
    src_lang = "eng_Latn"
    batch = ip.preprocess_batch(input_sentences, src_lang = src_lang, tgt_lang = tgt_lang)
    inputs = tokenizer_ind(batch, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        generated_tokens = model.generate(**inputs, use_cache=True, min_length=0, max_length=256, num_beams=5, num_return_sequences=1)
    
    with tokenizer_ind.as_target_tokenizer():
        generated_tokens = tokenizer_ind.batch_decode( generated_tokens.detach().cpu().tolist(),skip_special_tokens=True,clean_up_tokenization_spaces=True,)
    # Postprocess the translations, including entity replacement
    translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)
    for input_sentence, translation in zip(input_sentences, translations):
        print(f"{src_lang}: {input_sentence}")
        print(f"{tgt_lang}: {translation}")
        return translation

# Function for generating a caption using BLIP2 model
def gen_caption_blip(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt").to("cpu")
    pixel_values = inputs.pixel_values
    model = model_blip.to("cpu")
    generated_ids = model.generate(pixel_values=pixel_values, max_length=10)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption


# Gradio interface
def process_image(image, model_name, lang):
    save_dir = "uploaded_images"
    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, "uploaded_image.jpg")
    image.save(image_path)
    
    eng_caption = gen_caption_blip(image_path)
    
    if lang == "English":
        translated_caption = eng_caption
        lange ="en"
    else:
        translated_caption = translate(eng_caption, trans_dict[lang][0])
        lange = trans_dict[lang][1]

    audio_output = gTTS("Predicted Caption is: " + translated_caption, lang = lange,  slow = False)
    audio_output.save("audio.mp3")

    return eng_caption, translated_caption, "audio.mp3"

# Launch Gradio interface
interface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(label="Model used", value="Blip2 fine-tuned Model"),
        gr.Dropdown(label="Select Language", choices=["Bengali", "English", "Gujarati", "Hindi", "Kannada", "Malayalam", "Marathi", "Punjabi", "Tamil", "Telugu", "Urdu"], value="English"),
    ],
    outputs=[
        gr.Textbox(label="Generated English Caption"),
        gr.Textbox(label="Translated Caption"),
        gr.Audio(label="Audio Output"),
    ],
    examples=example_images,
    title="Eye for Blind",
    description="Upload an image, select language to get audio output for visually challeneged person."
)
# Launch the interface
interface.launch()

