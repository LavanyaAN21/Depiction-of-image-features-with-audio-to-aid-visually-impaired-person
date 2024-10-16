#Install necessary libraries run once in laptop, no need to run it again and again
!pip install transformers
!pip install datasets
!pip install torch
!pip install torchvision
!pip install PIL
!pip install tqdm

!kaggle datasets download -d adityajn105/flickr8k
/content/captions.txt


import zipfile

zip_path = '/content/flickr8k.zip'  # Update with the path where your flickr8k.zip is located
extract_path = '/content/'  # Update with the path where you want to extract the files

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)


from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, GPT2Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


#Define the model, tokenizer, and feature extractor
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")



#Load and preprocess images
def load_image(image_path):
    image = Image.open(image_path)
    return feature_extractor(images=image, return_tensors="pt").pixel_values

#Example image
image_path = '/content/Images/1009434119_febe49276a.jpg'  # Update with correct path
pixel_values = load_image(image_path)

#Generate caption
outputs = model.generate(pixel_values)
caption = tokenizer.decode(outputs[0], skip_special_tokens=True)


!pip install transformers

!pip install google-cloud-translate

#Function to load translation model and tokenizer for Kannada
def load_translation_model_kn():
    model_name_kn = 'Helsinki-NLP/opus-mt-en-kn'
    translation_model_kn = MarianMTModel.from_pretrained(model_name_kn)
    translation_tokenizer_kn = MarianTokenizer.from_pretrained(model_name_kn)
    return translation_model_kn, translation_tokenizer_kn


#Function to load translation model and tokenizer for Kannada (using an alternative model)
def load_translation_model_kn():
    model_name_kn = 'Helsinki-NLP/opus-mt-en-kn'  # Update with an accessible model name if available
    translation_model_kn = MarianMTModel.from_pretrained(model_name_kn)
    translation_tokenizer_kn = MarianTokenizer.from_pretrained(model_name_kn)
    return translation_model_kn, translation_tokenizer_kn


pip install googletrans==4.0.0-rc1


from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, GPT2Tokenizer
from PIL import Image
import matplotlib.pyplot as plt
from googletrans import Translator

#Function to translate text to Kannada using Google Translate
def translate_to_kannada(text):
    translator = Translator()
    translated = translator.translate(text, src='en', dest='kn')
    return translated.text

#Function to translate text to Hindi using Google Translate
def translate_to_hindi(text):
    translator = Translator()
    translated = translator.translate(text, src='en', dest='hi')
    return translated.text

#Function to load the image and generate caption in English
def generate_caption(image_path):
    caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    caption_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    #Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

    #Generate caption
    outputs = caption_model.generate(pixel_values, max_length=16, num_beams=4, early_stopping=True)
    caption = caption_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return caption

#Example image path
image_path = '/content/Images/1000268201_693b08cb0e.jpg'  # Update with correct path

#Generate caption in English
caption = generate_caption(image_path)
print(f"Caption in English: {caption}")

#Translate caption to Kannada
translated_caption_kn = translate_to_kannada(caption)
print(f"Caption in Kannada: {translated_caption_kn}")

#Translate caption to Hindi
translated_caption_hi = translate_to_hindi(caption)
print(f"Caption in Hindi: {translated_caption_hi}")

#Display the image
img = Image.open(image_path)
plt.imshow(img)
plt.axis('off')
plt.show()


pip install gtts


from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, GPT2Tokenizer
from PIL import Image
import matplotlib.pyplot as plt
from googletrans import Translator
from gtts import gTTS
import IPython.display as ipd

#Function to translate text to Kannada using Google Translate
def translate_to_kannada(text):
    translator = Translator()
    translated = translator.translate(text, src='en', dest='kn')
    return translated.text

#Function to translate text to Hindi using Google Translate
def translate_to_hindi(text):
    translator = Translator()
    translated = translator.translate(text, src='en', dest='hi')
    return translated.text

#Function to convert text to speech
def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save("caption.mp3")
    return "caption.mp3"

#Function to load the image and generate caption in English
def generate_caption(image_path):
    caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    caption_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    #Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

    #Generate caption
    outputs = caption_model.generate(pixel_values, max_length=16, num_beams=4, early_stopping=True)
    caption = caption_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return caption

#Example image path
image_path = '/content/Images/1000268201_693b08cb0e.jpg'  # Update with correct path

#Generate caption in English
caption = generate_caption(image_path)
print(f"Caption in English: {caption}")

#Translate caption to Kannada
translated_caption_kn = translate_to_kannada(caption)
print(f"Caption in Kannada: {translated_caption_kn}")

#Translate caption to Hindi
translated_caption_hi = translate_to_hindi(caption)
print(f"Caption in Hindi: {translated_caption_hi}")

#Convert captions to speech and play them
audio_en = text_to_speech(caption, lang='en')
ipd.display(ipd.Audio(audio_en))

audio_kn = text_to_speech(translated_caption_kn, lang='kn')
ipd.display(ipd.Audio(audio_kn))

audio_hi = text_to_speech(translated_caption_hi, lang='hi')
ipd.display(ipd.Audio(audio_hi))

#Display the image
img = Image.open(image_path)
plt.imshow(img)
plt.axis('off')
plt.show()
