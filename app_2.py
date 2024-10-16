# Import required libraries
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from gtts import gTTS
from IPython import display
import pickle
import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Blip2ForConditionalGeneration, Blip2Processor
from peft import PeftModel, PeftConfig
import IndicTransToolkit 
from IndicTransToolkit.IndicTransToolkit import IndicProcessor

# Setup paths and models
CHECKPOINT_PATH = os.path.join("Eye_blind_files", "train")
OPTIMIZER = tf.keras.optimizers.Adam()
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


# Encoder class
class Encoder(tf.keras.Model):
    def __init__(self, embed_dim):
        super(Encoder, self).__init__()
        self.dense = tf.keras.layers.Dense(embed_dim)

    def call(self, features):
        features = self.dense(features)
        features = tf.nn.relu(features)
        return features

# Attention model class
class AttentionModel(tf.keras.Model):
    def __init__(self, units):
        super(AttentionModel, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        self.units = units

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        attention_hidden_layer = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        score = self.V(attention_hidden_layer)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# Decoder class
class Decoder(tf.keras.Model):
    def __init__(self, embed_dim, units, vocab_size):
        super(Decoder, self).__init__()
        self.units = units
        self.attention = AttentionModel(self.units)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.gru = tf.keras.layers.GRU(self.units, return_sequences=True, return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = self.fc1(output)
        output = tf.reshape(output, (-1, output.shape[2]))
        output = self.fc2(output)
        return output, state, attention_weights

    def init_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

# Setup models and checkpoint manager
embedding_dim = 256
units = 512
vocab_size = 5001
encoder = Encoder(embedding_dim)
decoder = Decoder(embedding_dim, units, vocab_size)

ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=OPTIMIZER)
ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=5)

# Restore the latest checkpoint if available
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print(f'Model restored from checkpoint: {ckpt_manager.latest_checkpoint}')
else:
    print('No checkpoint found, please train the model first.')

# Load tokenizer
with open('Eye_blind_files/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_sequence_len = 35
attention_features_shape = 64

# Pretrained InceptionV3 model for image feature extraction
image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

# Utility functions
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

# Function for evaluation
def evaluate(image):
    attention_plot = np.zeros((max_sequence_len, attention_features_shape))
    hidden = decoder.init_state(batch_size=1)
    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    features = encoder(img_tensor_val)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_sequence_len):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()
        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot, predictions

        dec_input = tf.expand_dims([predicted_id], 0)
    
    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot, predictions

# Filter function for the generated text
def filter_text(text):
    remove_tokens = ['<start>', '<unk>', '<end>']
    tokens = text.split()
    return ' '.join([word for word in tokens if word not in remove_tokens])

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

# function to generate caption
def test_caption_generation(img_test):

    result, attention_plot,pred_test = evaluate(img_test)
    pred_caption=' '.join(result).rsplit(' ', 1)[0]

    print ('Prediction Caption:', pred_caption)
    # use Google Text to Speech Online API from playing the predicted caption as audio
    speech = gTTS("Predicted Caption is: "+ pred_caption,lang = 'en', slow = False)
    speech.save('audio.mp3')
    audio_file = 'audio.mp3'
    #playsound('voice.wav')
    display.display(display.Audio(audio_file, rate=None,autoplay=False))

    #return the test image and attention plot
    return result, attention_plot

# Function for generating a caption using BLIP2 model
def gen_caption_blip(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt").to("cpu")
    pixel_values = inputs.pixel_values
    model = model_blip.to("cpu")
    generated_ids = model.generate(pixel_values=pixel_values, max_length=10)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption

# function to plot attention map
def plot_attmap(caption, weights):
    weights_map=[]
    cap_map=[]
    
    len_cap = len(caption)
    for cap in range(len_cap):
        weights_img = np.reshape(weights[cap], (8,8))
        #reshape
        weights_img = np.array(Image.fromarray(weights_img).resize((224, 224), Image.LANCZOS))
        weights_map.append(weights_img)
        cap_map.append(caption[cap])
    plt.imshow(weights_map[0],cmap='gist_heat')
    return weights_map,cap_map

def get_attention_plot(result,attention_plot):
    output_images,image_titles = plot_attmap(result,attention_plot)
    plt.imshow(output_images[0])

    num_images = len(output_images)  # Number of random images to generate
    images = []  # List to hold images
   
    for i in range(num_images):
        image = output_images[i]  # Create a random RGB image
        plt.imshow(image,cmap='gist_heat', alpha =0.6)
        plt.axis('off')  # Turn off axis
        title = image_titles[i]  # Title for each image
        plt.title(title)

        # Save the figure to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the figure to prevent display
        buf.seek(0)  # Rewind the BytesIO object
        img = Image.open(buf)  # Open the image using PIL
        images.append(img)  # Append the PIL image to the list
    return images

def gen_cap(img_test_path):
    result, attention_plot,pred_test = evaluate(img_test_path)
    pred_caption=' '.join(result).rsplit(' ', 1)[0]
    #return the test image and attention plot
    return pred_caption,result,attention_plot

# Gradio interface
def process_image(image, model_name, lang):
    save_dir = "uploaded_images"
    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, "uploaded_image.jpg")
    image.save(image_path)
    
    if model_name == "Model from scratch":
        eng_caption, result, attention_plot = gen_cap(image_path)
    else:
        eng_caption = gen_caption_blip(image_path)
    
    if lang == "English":
        translated_caption = eng_caption
        lange ="en"
    else:
        translated_caption = translate(eng_caption, trans_dict[lang][0])
        lange = trans_dict[lang][1]

    audio_output = gTTS("Predicted Caption is: " + translated_caption, lang = lange,  slow = False)
    audio_output.save("audio.mp3")

    if model_name == "Model from scratch":
        attention_images = get_attention_plot(result, attention_plot)
        return eng_caption, translated_caption, "audio.mp3", gr.update(value = attention_images , visible=True)
    else:
        return eng_caption, translated_caption, "audio.mp3", gr.update(visible = False)

# Launch Gradio interface
interface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Dropdown(label="Select Model", choices=["Blip2 fine-tuned", "Model from scratch"], value="Blip2 fine-tuned"),
        gr.Dropdown(label="Select Language", choices=["Bengali", "English", "Gujarati", "Hindi", "Kannada", "Malayalam", "Marathi", "Oriya", "Punjabi", "Sanskrit", "Tamil", "Telugu", "Urdu"], value="English"),
    ],
    outputs=[
        gr.Textbox(label="Generated English Caption"),
        gr.Textbox(label="Translated Caption"),
        gr.Audio(label="Audio Output"),
        gr.Gallery(label="Attention Plots")
    ],
    examples=example_images,
    title="Eye for Blind",
    description="Upload an image, select model and language to get audio output for visually challeneged person."
)
# Launch the interface
interface.launch()

