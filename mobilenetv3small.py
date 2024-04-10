import keras
import tensorflow as tf
from keras.optimizers import SGD
from keras.applications.mobilenet_v3 import preprocess_input
from keras.applications import MobileNetV3Small
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Input, RandomFlip, RandomContrast, RandomBrightness


class LogCallback(keras.callbacks.Callback):
    def __init__(self,filename='train_log.log'):
        self.file = open(filename, 'a')
        self.file.write('[')

    def on_train_batch_end(self, batch, logs=None):
        self.file.write(f'{logs},\n'.replace("'",'"'))

    def on_train_end(self,logs=None):
        self.file.write('{}]')
        self.file.close()

print('Loading datasets...')
training_data, vali_data = keras.utils.image_dataset_from_directory(
        directory = './rscd/train/',
        label_mode = 'categorical',
        seed = 123,
        validation_split = 0.05,
        image_size=(240,360),
        subset = 'both')

test_data = keras.utils.image_dataset_from_directory(
        directory = './rscd/val',
        label_mode = 'categorical',
        image_size = (240,360)
        )

preprocessing = [
    RandomFlip('horizontal'),
    RandomContrast(0.3),
    RandomBrightness(0.3),
    preprocess_input
]

@tf.function
def training_preprocess(xs):
  for process in preprocessing:
    xs = process(xs)
  return xs

training_data = training_data.map(lambda xs,ys: (training_preprocess(xs),ys))
vali_data = vali_data.map(lambda xs,ys: (preprocess_input(xs),ys))
test_data = test_data.map(lambda xs,ys: (preprocess_input(xs),ys))

print('Setting up model...')
base_model = MobileNetV3Small(weights = 'imagenet', include_top = False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation = 'relu')(x)
x = Flatten()(x)
prediction_layer =  Dense(27, activation = 'softmax')(x)
model = Model(inputs = base_model.input, outputs = prediction_layer)
base_model.trainable = False
model.compile(optimizer = keras.optimizers.Adam(), 
        loss = 'categorical_crossentropy',
        metrics = [keras.metrics.CategoricalAccuracy(),
                   keras.metrics.TopKCategoricalAccuracy(5)])

print('Initial training...')

model.fit(training_data, 
        validation_data = vali_data,
        epochs = 4, 
        callbacks = [
            LogCallback('mobilenetv3small_detailed.json'),
            keras.callbacks.ModelCheckpoint(filepath='./tmp_mobilenetv3small/chck/{epoch:02d}_{categorical_accuracy:.8f}.keras', monitor='categorical_accuracy', save_freq=100),
            keras.callbacks.BackupAndRestore(backup_dir='./tmp_mobilenetv3small/backups', save_freq=100),
            keras.callbacks.CSVLogger('./mobilenetv3small.log')
            ])

model.trainable = True
model.save('initial_mobilenetv3small.keras')
model = keras.models.load_model('initial_mobilenetv3small.keras')

print('Fine-tuning...')
model.compile(optimizer=SGD(learning_rate = 0.0001, momentum = 0.9), 
              loss='categorical_crossentropy',
              metrics=[
                        keras.metrics.CategoricalAccuracy(),
                        keras.metrics.TopKCategoricalAccuracy(5)
                  ])
model.fit(training_data, validation_data = vali_data, epochs = 2,
            callbacks=[ LogCallback('mobilenetv3small_tuning_detailed.json'),
                        keras.callbacks.CSVLogger('./mobilenetv3small.log', append=True),
                        keras.callbacks.ModelCheckpoint(filepath='./tmp_mobilenetv3small/chck_tuning/{epoch:02d}_{categorical_accuracy:.8f}.keras', monitor='categorical_accuracy', save_freq=100),
                        keras.callbacks.BackupAndRestore(backup_dir='./tmp_mobilenetv3small/backups_tuning', save_freq=100)])
model.save('mobilenetv3small_gravel.keras')
model.evaluate(test_data, callbacks=[LogCallback('mobilenetv3small_test.json')])