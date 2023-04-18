import tensorflow as tf
import os
import numpy

images = os.path.join("images")
os.listdir(images)

raw_train_ds = tf.keras.utils.image_dataset_from_directory(
    directory = images,
    seed = 32,
    validation_split = 0.2,
    image_size = (224,224),
    subset="training"
)

raw_val_ds = tf.keras.utils.image_dataset_from_directory(
    directory = images,
    seed= 32,
    validation_split=0.2,
    image_size=(224,224),
    subset = "validation"
)
optimization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = raw_train_ds.map(lambda x,y : (optimization_layer(x),y))
val_ds = raw_val_ds.map(lambda x,y : (optimization_layer(x),y))
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(155, activation='softmax')
])
print("model processing done!")
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimization = tf.keras.optimizers.Adam()
model.compile(
    optimizer=optimization,
    loss=loss,
    metrics=["accuracy"]
)
history = model.fit(train_ds,epochs=30)
print("model traning done!")
model.evaluate(val_ds,verbose=10)
model.save("Dog_breed.h5")