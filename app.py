#Import all the required libraries

# tensorflow ml library
import tensorflow as tf

# libraries for data manipulation
import numpy as np

import os
# libraries for visualisation
import seaborn as sns
import matplotlib.pyplot as plt

# library for model build
from sklearn.model_selection import train_test_split

#library for image processing
from PIL import Image
from io import BytesIO

#audio libraries
from gtts import gTTS
from playsound import playsound
from IPython import display

import pickle

import gradio as gr
from gtts import gTTS

# Create the checkpoint path
checkpoint_path = os.path.join("/Users/gokulgopank/Documents/deploy/checkpoints", "train")
optimizer = tf.keras.optimizers.Adam()



import torch
from transformers import (AutoModelForSeq2SeqLM,AutoTokenizer)
import IndicTransToolkit 
from IndicTransToolkit import IndicProcessor

from transformers import Blip2ForConditionalGeneration
from peft import PeftModel, PeftConfig
from tqdm import tqdm
from transformers import Blip2Processor

peft_model_id = "/Users/gokulgopank/Documents/deploy/peft"
config = PeftConfig.from_pretrained(peft_model_id)
offload_folder = '/Users/gokulgopank/Documents/deploy/offload'
device = "cuda" if torch.cuda.is_available() else "cpu"
device_map = {
    "query_tokens": 0,
    "vision_model":0,
    "language_model": 1,
    "language_projection": 1,
    "lm_head": 1,
    "qformer": 0,
}
max_memory = {i: "4GB" for i in range(2)} #2 GPU

model_blip = Blip2ForConditionalGeneration.from_pretrained(config.base_model_name_or_path,
                                                        device_map = "auto",
                                                      torch_dtype = torch.float16,offload_folder=offload_folder
                                                     )

model_blip = PeftModel.from_pretrained(model_blip, peft_model_id)
checkpoint = 'Salesforce/blip2-opt-2.7b'
processor = Blip2Processor.from_pretrained(checkpoint)

#set embedding dimension
embedding_dim = 256
units = 512
#top 5,000 words +1
vocab_size = 5001

#create encoder class by subclass method
class Encoder(tf.keras.Model):
    def __init__(self,embed_dim):
        super(Encoder, self).__init__()
        #build your Dense layer with relu activation
        self.dense = tf.keras.layers.Dense(embed_dim)

    def call(self, features):
        # extract the features from the image shape: (batch, 8*8, embed_dim)
        features =  self.dense(features)
        #add relu activation
        features = tf.nn.relu(features)
        return features
    
#create encoder class by subclass method
class Attention_model(tf.keras.Model):
    def __init__(self, units):
        super(Attention_model, self).__init__()
        #build your Dense layers
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        #build your final Dense layer with unit 1
        self.V = tf.keras.layers.Dense(1)
        self.units=units

    def call(self, features, hidden):
        #features shape: (batch_size, 8*8, embedding_dim)
        # hidden shape: (batch_size, hidden_size)
        # Expand the hidden shape to shape: (batch_size, 1, hidden_size)
        hidden_with_time_axis =  tf.expand_dims(hidden,1)
        attention_hidden_layer = (tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)))
        # build your score funciton to shape: (batch_size, 8*8, units)
        score = self.V(attention_hidden_layer)
        # extract your attention weights with shape: (batch_size, 8*8, 1)
        attention_weights =  tf.nn.softmax(score,axis=1)
        #shape: create the context vector with shape (batch_size, 8*8,embedding_dim)
        context_vector =  attention_weights * features
        # reduce the shape to (batch_size, embedding_dim)
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights
    
class Decoder(tf.keras.Model):
    def __init__(self, embed_dim, units, vocab_size):
        super(Decoder, self).__init__()
        self.units=units
        #iniitalise Attention model with units
        self.attention = Attention_model(self.units)
        #build your Embedding layer
        self.embed = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # initialise GRU layer
        self.gru = tf.keras.layers.GRU(self.units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
        #build your Dense layer
        self.d1 = tf.keras.layers.Dense(self.units)
        #build your Dense layer with vocab dimension
        self.d2 = tf.keras.layers.Dense(vocab_size)


    def call(self,x,features, hidden):
        #create context vector & attention weights from attention model
        context_vector, attention_weights = self.attention(features, hidden)
        # embed input to shape: (batch_size, 1, embedding_dim)
        embed =  self.embed(x)
        # Concatenate input with the context vector from attention layer. Shape: (batch_size, 1, embedding_dim + embedding_dim)
        embed =  tf.concat([tf.expand_dims(context_vector, 1), embed], axis=-1)
        # Extract the output & hidden state from GRU layer. Output shape : (batch_size, max_length, hidden_size)
        output,state = self.gru(embed)
        output = self.d1(output)
        # shape : (batch_size * max_length, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))
        # shape : (batch_size * max_length, vocab_size)
        output = self.d2(output)
        return output,state, attention_weights

    def init_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
    

encoder=Encoder(embedding_dim)
decoder=Decoder(embedding_dim, units, vocab_size)

# Define the checkpoint object
ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# Restore the latest checkpoint
latest_ckpt = ckpt_manager.latest_checkpoint
if latest_ckpt:
    ckpt.restore(latest_ckpt)
    print(f'Model restored from checkpoint: {latest_ckpt}')
else:
    print('No checkpoint found, please train the model first.')


# Load the tokenizer from the saved file
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

#load pretrained model
image_model = tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')
#get the input of the image_model
new_input = image_model.input
#get the output of the image_model
hidden_layer = image_model.layers[-1].output
#build the final model using both input & output layer
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

def load_image(image_path):
    #read file
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    # resize to required shape
    img = tf.image.resize(img, (299, 299))
    # apply preprocess of inception 3 module
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

max_sequence_len = 35

attention_features_shape = 64
def evaluate(image):
    #initilaise decoder
    attention_plot = np.zeros((max_sequence_len, attention_features_shape))
    hidden = decoder.init_state(batch_size=1)
    #process the input image to desired format before extracting features
    print(image)
    temp_input = tf.expand_dims(load_image(image)[0],0)
    # Extract features using our feature extraction model
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    # extract the features by passing the input to encoder
    features = encoder(img_tensor_val)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_sequence_len):
        # get the output from decoder
        predictions, hidden, attention_weights = decoder(dec_input,features,hidden)
        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
        #extract the predicted id(embedded value) which carries the max value
        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])
        #map the id to the word from tokenizer and append the value to the result list
        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot,predictions

        dec_input = tf.expand_dims([predicted_id], 0)
    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot,predictions

#filter the caption
def filt_text(text):
    filt=['<start>','<unk>','<end>']
    temp= text.split()
    # remove uncessary tokens
    [temp.remove(j) for k in filt for j in temp if k==j]
    text=' '.join(temp)
    return text

# function to generate caption
def test_caption_generation(img_test):

    result, attention_plot,pred_test = evaluate(img_test)
    pred_caption=' '.join(result).rsplit(' ', 1)[0]

    #candidate = pred_caption.split()
    print ('Prediction Caption:', pred_caption)
    # use Google Text to Speech Online API from playing the predicted caption as audio
    speech = gTTS("Predicted Caption is: "+ pred_caption,lang = 'en', slow = False)
    speech.save('audio.mp3')
    audio_file = 'audio.mp3'
    #playsound('voice.wav')
    display.display(display.Audio(audio_file, rate=None,autoplay=False))

    #return the test image and attention plot
    return result, attention_plot

model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
tokenizer_ind = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)

ip = IndicProcessor(inference=True)

def gen_cap(img_test_path):
    result, attention_plot,pred_test = evaluate(img_test_path)
    pred_caption=' '.join(result).rsplit(' ', 1)[0]
    #return the test image and attention plot
    return pred_caption,result,attention_plot


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

def translate(eng_cap,tgt_lang):
    input_sentences = [eng_cap]
    src_lang = "eng_Latn"
    batch = ip.preprocess_batch(input_sentences,src_lang=src_lang,tgt_lang=tgt_lang,)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Tokenize the sentences and generate input encodings
    inputs = tokenizer_ind(batch,truncation=True,padding="longest",return_tensors="pt",return_attention_mask=True,).to(DEVICE)
# Generate translations using the model
    with torch.no_grad():generated_tokens = model.generate(**inputs, use_cache=True,min_length=0,max_length=256,num_beams=5, num_return_sequences=1,)
# Decode the generated tokens into text
    with tokenizer_ind.as_target_tokenizer():generated_tokens = tokenizer_ind.batch_decode( generated_tokens.detach().cpu().tolist(),skip_special_tokens=True,clean_up_tokenization_spaces=True,)
# Postprocess the translations, including entity replacement
    translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)
    for input_sentence, translation in zip(input_sentences, translations):
        print(f"{src_lang}: {input_sentence}")
        print(f"{tgt_lang}: {translation}")
        return translation
    
trans_dict = {"Bengali": ["ben_Beng","bn"],"English": ["eng_Latn","en"],"Gujarati": ["guj_Gujr","gu"],
    "Hindi": ["hin_Deva","hi"],"Kannada": ["kan_Knda","kn"],"Malayalam": ["mal_Mlym","ml"],"Marathi": ["mar_Deva","mr"],
    "Nepali": ["npi_Deva","ne"],"Punjabi": ["pan_Guru","pa"],"Tamil": ["tam_Taml","ta"],"Telugu": ["tel_Telu","te"],"Urdu": ["urd_Arab","ur"],}

example_images = [["/Users/gokulgopank/Documents/deploy/Capstone/Images/667626_18933d713e.jpg"],["/Users/gokulgopank/Documents/deploy/Capstone/Images/3637013_c675de7705.jpg"]]

def get_att_plot(result,attention_plot):
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

def gen_cap_blip(image_path):
    image = Image.open(image_path)

    #Inferance on CPU
    inputs = processor(images = image, return_tensors = "pt").to("cpu")
    pixel_values = inputs.pixel_values

    model = model_blip.to("cpu")
    generated_ids = model.generate(pixel_values = pixel_values, max_length = 10)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print('Caption: '+ generated_caption)
    return generated_caption

def process_image(image,mod_name, lan):
    global aa
    save_dir = "uploaded_images"
    os.makedirs(save_dir, exist_ok=True)

    # Save the uploaded image
    image_path = os.path.join(save_dir, "uploaded_image.jpg")
    image.save(image_path)
    if mod_name == "Model from scratch":
        eng_cap ,result,attention_plot= gen_cap(image_path)
    else:
        eng_cap = gen_cap_blip(image_path)
    # Dummy description generation for demonstration
    if lan == "English":
        tar_cap = eng_cap
        lange ="en"
    else:
        tar_cap = translate(eng_cap,trans_dict[lan][0])
        lange = trans_dict[lan][1]

    # Convert text to speech
    speech = gTTS("Predicted Caption is: "+ tar_cap,lang = lange, slow = False)
    speech.save('audio.mp3')
    if mod_name == "Model from scratch":
        message = "Using a model trained from scratch. Results may vary."
    else:
        message = "Using blip2 fine tuned model. Inference may take upto 30s"
    if mod_name == "Model from scratch":
        images = get_att_plot(result,attention_plot)   
        return message, eng_cap, tar_cap, "audio.mp3", gr.update(value=images, visible=True)
    else:
        return message, eng_cap, tar_cap, "audio.mp3", gr.update(visible=False)


# Create the Gradio interface
interface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Dropdown(
            label="Select model",
            choices=["Blip2 fine-tuned", "Model from scratch"],
            value="Model from scratch",
        ),
        gr.Dropdown(
            label="Select Language",
            choices=["Bengali", "English", "Gujarati", "Hindi", "Kannada", "Malayalam", "Marathi", "Nepali", "Punjabi", "Tamil", "Telugu", "Urdu"],
            value="English",
        )
    ],
    outputs=[
        gr.Textbox(label="Message"),
        gr.Textbox(label="English caption "),
        gr.Textbox(label="Caption in preferred language"),
        gr.Audio(label="Audio Description"),
        # gr.Gallery(label="Generated Images", show_label=True, elem_id="gallery")
        gr.Gallery(label="Random Images"),     
    ],
    examples=example_images,
    title="Image to Text and Audio",
    description="Upload an image, select a category, and get a description in text and audio format."
)

# Launch the interface
interface.launch()
 
    
