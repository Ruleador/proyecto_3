import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

train_carpeta = "dataset/"

datagen = ImageDataGenerator(
    rescale =1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
)

train_data = datagen.flow_from_directory(
    train_carpeta,
    target_size=(224,224),
    batch_size=32,
    subset="training"
)

val_data = datagen.flow_from_directory(
    train_carpeta,
    target_size=(224, 224),
    batch_size=32,
    subset="validation"
)


labels = list(train_data.class_indices.keys())
with open("labels.txt", "w") as f:
    for label in labels:
        f.write(label + "\n")

modelo_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
modelo_base.trainable = False

x = modelo_base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
output = layers.Dense(len(labels), activation='softmax')(x)

modelo = models.Model(inputs=modelo_base.input, outputs=output)

modelo.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenamiento
modelo.fit(train_data, validation_data=val_data, epochs=5)

# Guardar modelo
modelo.save("modelo_perros.h5")