import tensorflow as tf
import keras
import sys
import os
import json
import pathlib

if len(sys.argv) <= 1:
    print(f'Usage: {sys.argv[0]} <models>')

preprocess = {
            'mobilenetv3small': keras.applications.mobilenet_v3.preprocess_input,
            'convnexttiny': keras.applications.convnext.preprocess_input,
            'inceptionv3': keras.applications.inception_v3.preprocess_input,
            'efficientnet': keras.applications.efficientnet.preprocess_input,
            'efficientnetv2': keras.applications.efficientnet_v2.preprocess_input
        }

print('Processing datasets...')
classes = list(enumerate(sorted(next(iter(os.walk('./rscd/test')))[1])))
with tf.device('/gpu:0'):
    datasets = []
    for i, label in classes:
        img_dataset = keras.utils.image_dataset_from_directory(f'./rscd/test/{label}',shuffle=False, labels=None, image_size=(240,360))
        datasets.append({'label': label,'unlabeled': img_dataset, 'data': img_dataset.map(lambda x: (x,tf.map_fn(lambda y: tf.one_hot(tf.constant(i),27),x)))})
    datasets.append({'label': 'overall', 'data': keras.utils.image_dataset_from_directory('./rscd/test',shuffle=False, label_mode='categorical', image_size=(240,360))})

print('Loading models...')
models = []
for model_filename in sys.argv[1:]:
    models.append((model_filename,keras.saving.load_model(model_filename,compile=False)))

print('Evaluating...')
for model_filename, model in models:
    curr_hist = []
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy', 'top_k_categorical_accuracy'])
    preprocess_input = preprocess[pathlib.Path(model_filename).stem.split('_')[0]]
    for dataset in datasets:
        ds = dataset['data'].map(lambda x,y: (preprocess_input(x),y)).unbatch().map(lambda u,v: (tf.expand_dims(u,0),tf.expand_dims(v,0)))
        options = tf.data.Options()
        options.deterministic = True
        ds = ds.with_options(options)
        ys = tf.constant(list(ds.map(lambda x,y: tf.cast(tf.vectorized_map(tf.argmax,y),tf.float32)).as_numpy_iterator()))
        tf.print(ys)
        with tf.device('/gpu:0'):
            res = model.evaluate(ds)
            xs = tf.cast(tf.vectorized_map(tf.argmax,model.predict(ds.map(lambda x,y:x))),tf.float32)

        epitaph = tf.math.confusion_matrix(ys,xs,27)
        curr_hist.append({'label': dataset['label'],'epitaph': epitaph.numpy().tolist(), 'results':{'loss': res[0], 'categorical_accuracy': res[1], 'top_5_accuracy': res[2]}})
        
    with open(model_filename[:-len('.keras')]+'.json', 'w') as log_file:
        json.dump(curr_hist, log_file)

