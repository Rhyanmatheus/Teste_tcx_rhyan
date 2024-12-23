from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Caminhos para os dados de treinamento e teste
train_dir = 'frutas_escolhidas/Training'
test_dir = 'frutas_escolhidas/Test'

# Data Augmentation para treinamento e normalização para teste
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Carregar dados
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical'
)

# Verificar se o modelo já existe
try:
    model = load_model('modelo_frutas_treinado.h5')
    print("Modelo carregado com sucesso!")
except:
    print("Modelo não encontrado! Treinando um novo modelo...")

    # Transfer Learning com MobileNetV2
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
    base_model.trainable = False  # Congela os pesos da base pré-treinada

    # Construir o modelo
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(3, activation='softmax')  # 3 classes
    ])

    # Compilar o modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Treinar o modelo
    history = model.fit(train_data, epochs=10, validation_data=test_data)

    # Salvar o modelo treinado
    model.save('modelo_frutas_treinado.h5')

    # Visualizar histórico de treinamento
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.title('Acurácia')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Perda de Treinamento')
    plt.plot(history.history['val_loss'], label='Perda de Validação')
    plt.title('Perda')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()
    plt.show()

# Avaliação do modelo
loss, accuracy = model.evaluate(test_data)
print(f"Acurácia do modelo nos dados de teste: {accuracy * 100:.2f}%")
