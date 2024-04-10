import sys
import sklearn
import os
import json
import tensorflow as tf
import numpy as np
import keras

class LogCallback(keras.callbacks.Callback):
    def __init__(self,filename='train_log.log'):
        self.hist = []
        try:
            if os.path.isfile(filename):
                with open(filename) as file:
                    self.hist = json.load(file)
        except:
            pass

        self.file = open(filename,'w')
    
    def on_train_batch_end(self, batch, logs=None):
        if logs:
            self.hist.append(logs)

    def on_train_epoch_end(self,_,logs=None):
        json.dump(self.hist, self.file)

class ModelList:
    def __init__(self, filenames, parser=None):
        self._models = self._load_models(filenames, filename_parser=parser)
        self._augmentations = [keras.layers.RandomFlip('horizontal'),
                                keras.layers.RandomContrast(0.3),
                                keras.layers.RandomBrightness(0.3)]

    @tf.function
    def mobilenetv3small(self, x,y):
        x = keras.applications.mobilenet_v3.preprocess_input(x)
        for augmentation in self._augmentations:
            x = augmentation(x)
        return x,y

    @tf.function
    def inceptionv3(self, x,y):
        x = keras.applications.inception_v3.preprocess_input(x)
        for augmentation in self._augmentations:
            x = augmentation(x)
        return x,y


    @tf.function
    def efficientnet(self, x,y):
        x = keras.applications.efficientnet.preprocess_input(x)
        for augmentation in self._augmentations:
            x = augmentation(x)
        return x,y


    @tf.function
    def efficientnetv2(self, x,y):
        x = keras.applications.efficientnet_v2.preprocess_input(x)
        for augmentation in self._augmentations:
            x = augmentation(x)
        return x,y

    @tf.function
    def convnexttiny(self, x,y):
        x = keras.applications.convnext.preprocess_input(x)
        for augmentation in self._augmentations:
            x = augmentation(x)
        return x,y


    def _load_models(self, filenames, filename_parser=None):
        if not filename_parser:
            filename_parser = self._parse_filename
        models = []
        for filename in filenames:
            models.append((filename_parser(filename),keras.models.load_model(filename)))
        return models
    
    def _parse_filename(self, filename):
        return filename.split('_')[0]

    def fine_tune(self, training, validation, epochs = 3, optimizer=keras.optimizers.SGD(learning_rate = 0.0001, momentum=0.9), loss = keras.losses.CategoricalFocalCrossentropy()):
        for model_name, model in self._models:
            print(f'============== {model_name} ==============')
            model.trainable = True
            for layer in model.layers[-(len(model.layers)//3):]:
                if not isinstance(layer, keras.layers.BatchNormalization):
                    layer.trainable = True
            for layer in model.layers[:-(len(model.layers)//3)]:
                layer.trainable = False

            model.compile(optimizer, loss = loss,
                                    metrics = [keras.metrics.CategoricalAccuracy(),
                                        keras.metrics.TopKCategoricalAccuracy(5)])

            preprocess_fn = getattr(self,model_name)
            train = training.map(preprocess_fn).prefetch(5)
            vali = validation.map(preprocess_fn).prefetch(5)
            clbk = LogCallback(f'{model_name}_improvements.json')
            try:
                model.fit(train, validation_data = vali,
                        epochs = epochs,
                            callbacks = [
                                clbk,
                                keras.callbacks.CSVLogger(f'./{model_name}_improve.log', append=True),
                                keras.callbacks.ModelCheckpoint(filepath=f'./improve_{model_name}/chck/{"{epoch:02d}"}_{"{categorical_accuracy:.8f}"}.keras', monitor='categorical_accuracy', save_freq=1000),
                                keras.callbacks.BackupAndRestore(backup_dir=f'./improve_{model_name}/backups', save_freq=100)
                                ])
            except:
                clbk.on_train_epoch_end(0)
            model.save(f'{model_name}_improved.keras')

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print(f'Usage: {sys.argv[0]} <models>')
        exit()
#    parse = lambda x: x.split('.')[0].split('_')[1]
    model_list = ModelList(sys.argv[1:])
    training_data = keras.utils.image_dataset_from_directory(
        directory = './rscd/train/',
        label_mode = 'categorical',
        batch_size = 64,
        image_size=(240,360))
    validation_data = keras.utils.image_dataset_from_directory(
        directory = './rscd/val/',
        label_mode = 'categorical',
        batch_size = 64,
        image_size=(240,360))

    with open('subdirs.json') as f:
        raw_class_sizes = json.load(f)

    class_sizes = dict(filter(lambda x: x[0] != 'overall', raw_class_sizes.items()))
    train_reconstruction = np.array([b for a in [[i]*v for i,(k,v) in enumerate(sorted(class_sizes.items()))] for b in a])
    class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(np.arange(27)),y=train_reconstruction)
    print(class_weights)
    loss_fn = keras.losses.CategoricalFocalCrossentropy(alpha=class_weights)
    optimizer = keras.optimizers.Adam(learning_rate = keras.optimizers.schedules.ExponentialDecay(0.0001,decay_rate=0.8,decay_steps=training_data.cardinality(),staircase=True))

    model_list.fine_tune(training_data, validation_data, epochs = 8, optimizer=optimizer, loss=loss_fn)
