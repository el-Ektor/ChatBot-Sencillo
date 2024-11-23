import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Inicializa el lematizador
lemmatizer = WordNetLemmatizer()

# Cargar el archivo de intenciones
with open("intents.json") as file:
    data = json.load(file)

# Preparar los datos
patterns = []
labels = []
responses = {}
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokenized_pattern = nltk.word_tokenize(pattern)
        patterns.append(tokenized_pattern)
        labels.append(intent["tag"])
    responses[intent["tag"]] = intent["responses"]

# Lematización de las palabras
words = [lemmatizer.lemmatize(w.lower()) for p in patterns for w in p]
words = sorted(list(set(words)))

# Codificar etiquetas
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)

# Convertir palabras a vectores de características (bag of words)
training = []
output_empty = [0] * len(data["intents"])

for x, pattern in enumerate(patterns):
    bag = []
    token_words = [lemmatizer.lemmatize(w.lower()) for w in pattern]
    for w in words:
        bag.append(1) if w in token_words else bag.append(0)
    training.append(bag)

X = np.array(training)
y = np.array(labels)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Entrenar modelo Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Guardar el modelo y las palabras
import pickle
with open("chatbot_model.pkl", "wb") as f:
    pickle.dump((model, words, encoder, responses), f)

print("Modelo entrenado y guardado")
