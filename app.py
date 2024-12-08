from flask import Flask, request, jsonify, send_from_directory
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import webbrowser
from flask_cors import CORS  # Importation de flask_cors

# Créer une instance Flask
app = Flask(__name__)

# Appliquer CORS à l'application Flask
CORS(app)  # Cela permet d'accepter des requêtes depuis n'importe quelle origine

# Charger le modèle (ajustez le chemin du modèle)
MODEL_PATH = "model_mobilenetV2.h5"
model = load_model(MODEL_PATH)

# Charger le fichier de mappage des classes vers les noms de poissons
def load_class_names(filepath):
    with open(filepath, 'r') as file:
        class_names = file.readlines()
    class_names = [name.strip() for name in class_names]  # Supprimer les sauts de ligne
    return class_names

# Charger les noms des poissons depuis le fichier
class_names = load_class_names("class_labels.txt")

# Prétraitement des images
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)  # Charger et redimensionner l'image
    img_array = img_to_array(img)  # Convertir en tableau numpy
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension batch
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)  # Prétraitement
    return img_array

# Route pour servir le fichier index.html
@app.route('/')
def home():
    return send_from_directory(os.getcwd(), 'index.html')

# Route pour effectuer une prédiction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Sauvegarder l'image temporairement
    filepath = os.path.join("temp", file.filename)
    file.save(filepath)

    # Prétraitement et prédiction
    img_array = preprocess_image(filepath)
    predictions = model.predict(img_array)
    os.remove(filepath)  # Supprimer le fichier temporaire

    # Obtenir le nom de la classe prédite à partir des résultats
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]
    confidence = np.max(predictions)

    return jsonify({
        "predicted_class": predicted_class_name,
        "confidence": float(confidence)
    })

# Lancer le serveur Flask et ouvrir index.html dans le navigateur
if __name__ == '__main__':
    # Créer un dossier temporaire pour stocker les fichiers si nécessaire
    if not os.path.exists("temp"):
        os.makedirs("temp")
    
    # Ouvrir le fichier index.html dans le navigateur
    # webbrowser.open("http://127.0.0.1:5000/")

    # Lancer le serveur Flask
    app.run(debug=True, host='0.0.0.0', port=5000)

