## How to run app interface
* Creare virtual environment
  
```python -m venv .venv ```  
```source .venv/bin/activate ```

* Clone the repo

```git clone https://github.com/gokul-gopan-k/Eye-for-Blind.git```

```cd Eye-for-Blind```

* Make the script executable:
  
```chmod +x script.sh```

* Run the script:
  
```./script.sh```

* Run the app
  
```python app.py```


# Problem Statement

Today, in the world of social media, millions of images are uploaded daily. Some of them are about your friends and family, while some of them are about nature and its beauty. Imagine a condition where you are not able to see and enjoy these images — a problem that blind people face on a daily basis. According to the World Health Organization (WHO), it has been reported that there are around 285 million visually impaired people worldwide and out of these 285 million, 39 million are totally blind. 


# Model Pipeline

1) Encoder-decoder model
   Convert the image to text description first and then using a simple text to speech API, the extracted text description/caption will be converted to audio. So the central part of this capstone is focused on building the caption/text description as the second part, which is transforming the text to speech is relatively easy with the text to speech.

2) Blip-2 fine tuned model
   Blip-2 transformer model is fine tuned using LoRA (Low rank adoption) approach on the custom Flikr8 dataset.
   IndicTrans model is used to convert the text ouput into different indian regional languages before converting it to audio format.
