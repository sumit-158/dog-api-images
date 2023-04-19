# import tensorflow as tf
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# import os

# # Set up data generators
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=30,
#     horizontal_flip=True,
#     zoom_range=0.2,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.1,
#     fill_mode='nearest'
# )

# val_datagen = ImageDataGenerator(
#     rescale=1./255
# )
# images = os.path.join("images")
# os.listdir(images)

# train_generator = train_datagen.flow_from_directory(
#     "images",
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical'
# )

# val_generator = val_datagen.flow_from_directory(
#     "images",
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical'
# )

# Set up model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = base_model.output
print(x)
# x = GlobalAveragePooling2D()(x)
# predictions = Dense(152, activation='softmax')(x)

# model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# # Compile model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Set up callbacks
# early_stop = EarlyStopping(patience=3, monitor='val_accuracy', mode='max')
# model_checkpoint = ModelCheckpoint('/path/to/save/best/model.h5', save_best_only=True, monitor='val_accuracy', mode='max')

# # Train model
# history = model.fit(
#     train_generator,
#     epochs=10,
#     validation_data=val_generator,
#     callbacks=[early_stop, model_checkpoint]
# )

# # Evaluate model
# test_datagen = ImageDataGenerator(rescale=1./255)

# test_generator = test_datagen.flow_from_directory(
#     '/path/to/test/directory',
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical'
# )

# loss, accuracy = model.evaluate(test_generator)

# print('Test loss:', loss)
# print('Test accuracy:', accuracy)
