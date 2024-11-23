import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import json
import pickle
import random

lemmatizer = WordNetLemmatizer()

# Cargar el modelo entrenado y los datos
with open("chatbot_model.pkl", "rb") as f:
    model, words, encoder, responses = pickle.load(f)

# Función para procesar la entrada del usuario
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Convertir la frase en bag of words
def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Obtener una respuesta del chatbot
def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict([bow])[0]
    return encoder.inverse_transform([res])[0]

def get_response(tag):
    return random.choice(responses[tag])

# Chat
print("¡El chatbot está listo! Escribe algo para hablar con él.")

while True:
    message = input("Tú: ")
    if message.lower() == "salir":
        print("---Cerrando sesion---")
        break
    tag = predict_class(message)
    response = get_response(tag)
    print(f"Chatbot: {response}")
