import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import keras
import tensorflow as tf
import sys

if len(sys.argv) != 2:
    print(f'Usage: {sys.argv[0]} <model>')
    exit()

dataset_augmentations = [
                            keras.layers.RandomFlip('horizontal'),
                            keras.layers.RandomBrightness(0.3),
                            keras.layers.RandomContrast(0.3),
                        ]

@tf.function
def augment(x,y):
    for process in dataset_augmentations:
        x = process(x)
    return x,y


print('Setting up datasets...')
training, validation = keras.utils.image_dataset_from_directory('../rscd/train',
                                                                seed=321,
                                                                label_mode='categorical',
                                                                validation_split=0.05,
                                                                subset='both',
                                                                image_size=(240,360))
training = training.map(augment)
validation = validation.map(augment)

test = keras.utils.image_dataset_from_directory('../rscd/val',
                                                label_mode = 'categorical',
                                                image_size=(240,360))

test = test.map(augment)

print('Loading model...')
model = keras.models.load_model(sys.argv[1])
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.0001,momentum=0.9),loss='categorical_crossentropy',metrics=['accuracy'])
model.trainable = True

print('Fine-tuning...')
model.fit(training, validation_data = validation, epochs = 2)
model.save('fine_tuned.keras')

print('Done!')
model.evaluate(test)
