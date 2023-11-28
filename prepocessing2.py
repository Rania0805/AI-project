
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, EarlyStopping
import os
import matplotlib.pyplot as plt
import numpy as np

# Define the list of species you want to classify
species_list = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']

# Directory containing raw images
base_dir = r'C:\Users\Rania\Documents\Fall project 2023\Topic in intellligent Systems'
image_dir = os.path.join(base_dir, 'raw-img')

# Define the ImageDataGenerator object
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=40, 
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Define the batch size
batch_size = 32

# Define the train generator
train_generator = datagen.flow_from_directory(
    image_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='sparse',
    subset='training')

# Define the validation generator
validation_generator = datagen.flow_from_directory(
    image_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='sparse',
    subset='validation')

# Define the model
# Create a Sequential model, which represents a linear stack of layers
model = Sequential()

# Add a 2D convolutional layer with 32 filters,
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))

# Add a max-pooling layer with pool size (2, 2) to reduce spatial dimensions
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add batch normalization to normalize the activations of the previous layer
model.add(BatchNormalization())

# Add another 2D convolutional layer with 64 filters, each of size (3, 3), using ReLU activation
model.add(Conv2D(64, (3, 3), activation='relu'))

# Add another max-pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add batch normalization
model.add(BatchNormalization())

# Add a third 2D convolutional layer with 128 filters, each of size (3, 3), using ReLU activation
model.add(Conv2D(128, (3, 3), activation='relu'))

# Add another max-pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add batch normalization
model.add(BatchNormalization())

# Flatten the output to a one-dimensional array before fully connected layers
model.add(Flatten())

# Add a fully connected layer with 256 neurons using ReLU activation
model.add(Dense(256, activation='relu'))

# Add dropout layer with dropout rate of 0.5 for regularization
model.add(Dropout(0.5))

# Add the output layer with units equal to the number of classes and softmax activation for multi-class classification
model.add(Dense(len(species_list), activation='softmax'))


# Compile the model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define the learning rate schedule
def lr_schedule(epoch):
    lr = 0.0001
    if epoch > 10:
        lr *= 0.1
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)


early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=30,
    callbacks=[lr_scheduler, early_stopping])  # Use learning rate schedule and early stopping

# Visualize images with predictions
def visualize_images(generator, num_images=5):
    images, labels = generator.next()

    predictions = model.predict(images)

    # Convert numerical labels back to original species names
    predicted_species = [species_list[np.argmax(pred)] for pred in predictions]

    # Display images with predictions
    for i in range(num_images):
        image = images[i]
        label = species_list[int(labels[i])]
        predicted_label = predicted_species[i]

        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.title(f"True label: {label}\nPredicted label: {predicted_label}")
        plt.axis('off')
        plt.show()


visualize_images(train_generator)

visualize_images(validation_generator)

# Evaluate the model
loss, accuracy = model.evaluate_generator(validation_generator, steps=validation_generator.samples // batch_size)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

# Make predictions
predictions = model.predict_generator(validation_generator, steps=validation_generator.samples // batch_size)
predicted_species = [species_list[np.argmax(pred)] for pred in predictions]
print(predicted_species)
