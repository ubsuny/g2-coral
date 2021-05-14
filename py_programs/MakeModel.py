# This python file is for easily operating/ creating a CNN model

import tensorflow as tf
from tensorflow import keras
#import sys
#sys.path.append('./../py_programs/')
from keract import get_activations


# create a model
def make_model(input_shape):
    '''
    input_shape -> input data shape 
    return: a keras model
    '''
    # input shape should be (vinvalues, 1)
    input_layer = keras.layers.Input(input_shape)
    
    lam1 = keras.layers.Lambda(lambda x: -x)(input_layer)

    min1 = keras.layers.MaxPool1D(pool_size=4,strides=5, padding='same')(lam1)

    lam2 = keras.layers.Lambda(lambda x: -x)(min1)

    conv1 = keras.layers.Conv1D(filters=4, kernel_size=3, padding='same', strides=1)(lam2)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)
    
    # maxpooling layer
    pool = keras.layers.MaxPool1D(pool_size=5, strides=5, padding='same')(conv1) # keras.layers.GlobalAveragePooling1D()(conv3) #  # 

    # flatten layer
    flat = keras.layers.Flatten()(pool)

    #pool = keras.layers.GlobalAveragePooling1D()(conv1) # keras.layers.GlobalAveragePooling1D()(conv3) #  # 


    # fully connected layer to output a binary vector
    dense1 = keras.layers.Dense(10, activation='relu')(flat)
    dense2 = keras.layers.Dense(1, activation='sigmoid')(dense1)

    return keras.models.Model(inputs=input_layer, outputs=dense2)


# set a callback list
def set_callback(save=False, name='model'):
    '''
    save -> True for saving the model, by default it's False
    name -> set the file name
    return: the callback list
    '''
    if save:
        callbacks = [# save checkpoints
            keras.callbacks.ModelCheckpoint(name+".h5", save_best_only=True, monitor="val_loss"),
            # if there's no improvement for minimizing losses, which makes the training stagnate then reduce the learning rate
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001),
            # stop training if a monitored metric stops improving
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),]
    else:
        callbacks = [
            # if there's no improvement for minimizing losses, which makes the training stagnate then reduce the learning rate
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001),
            # stop training if a monitored metric stops improving
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),]
    return callbacks


# compile the model
def model_compile(model):
    '''
    input a keras model
    return the compile info
    '''
    model.compile(
    optimizer='adam',  #tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    loss='mse',
    metrics=['sparse_categorical_accuracy','binary_accuracy'],
    )
    return model.compile


# train the model
def model_train(model, x_train, y_train, batch_size, epochs, callbacks):
    '''
    model -> a keras model
    x_train -> training data
    y_train -> training data result
    batch_size -> batch size
    epochs -> epochs
    callbacks -> a callback list
    '''
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.3,
        verbose=1,
    )


# use keract to visualize layers
def visualize_layer(model, inputdata):
    '''
    model -> a keras model
    inputdata -> any data for the model input
    return:  layer output info
    '''
    activations = get_activations(model, inputdata, auto_compile=False)
    return [(k, '->', v.shape, '- Numpy array') for (k, v) in activations.items()]  , activations  