import keras
from keras.optimizers import SGD
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Input

print('Loading datasets...')
training_data, vali_data = keras.utils.image_dataset_from_directory(
        directory = './train/',
        label_mode = 'categorical',
        seed = 123,
        validation_split = 0.05,
        image_size=(240,360),
        subset = 'both')


print('Setting up model...')
base_model = InceptionV3(input_tensor = Input(shape=(240,360,3)), weights = 'imagenet', include_top = False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation = 'relu')(x)
x = Flatten()(x)
x = Dense(128, activation = 'relu')(x)
prediction_layer =  Dense(27, activation = 'softmax')(x)
model = Model(inputs = base_model.input, outputs = prediction_layer)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy')

print('Initial training...')

model.fit(training_data, validation_data = vali_data, epochs = 2)


for layer in model.layers[:-6]:
    layer.trainable = False

for layer in model.layers[-6:]:
    layer.trainable = True

print('Fine-tuning...')
model.compile(optimizer=SGD(learning_rate = 0.0001, momentum = 0.9), loss='categorical_crossentropy')
model.fit(training_data, validation_data = vali_data, epochs = 6)
model.save('inception_gravel.keras')
