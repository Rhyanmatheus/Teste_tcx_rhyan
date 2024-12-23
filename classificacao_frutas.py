import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Carregar o modelo treinado
model = load_model('modelo_frutas_treinado.h5')

# Dicionário de classes
class_dict = {0: 'Maçã', 1: 'Banana', 2: 'Morango'}

# Caminho para a pasta com imagens de teste
test_images_dir = 'test_images'

# Processar e fazer predições para cada imagem na pasta
for image_file in os.listdir(test_images_dir):
    # Carregar e pré-processar a imagem
    img_path = os.path.join(test_images_dir, image_file)
    img = load_img(img_path, target_size=(100, 100))  # Redimensionar para o tamanho esperado pelo modelo
    img_array = img_to_array(img) / 255.0  # Normalizar os valores dos pixels
    img_array = np.expand_dims(img_array, axis=0)  # Adicionar uma dimensão para lotes

    # Fazer a predição
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    accuracy = np.max(predictions) * 100

    # Exibir o resultado
    print(f"Imagem: {image_file}")
    print(f"Classe Predita: {class_dict[predicted_class]}")
    print(f"Acurácia: {accuracy:.2f}%")
    print("-" * 30)



