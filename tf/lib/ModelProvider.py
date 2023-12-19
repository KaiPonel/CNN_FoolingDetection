import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def get_model(model_type="cnn", dataset="mnist"):
    if model_type not in ["cnn", "mlp", "imagenet"]:
        raise Exception("Invalid value for `model_type`. Got {}, expected [`cnn`, `mlp`, `imagenet`]".format(model_type))
    if dataset not in ["mnist", "cifar10", "imagenet"]:
        raise Exception("Invalid value for `dataset`. Got {}, expected [`mnist`, `cifar10`, `imagenet`]".format(dataset))
    
    # Model Specific Hyperparameters
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.losses.SparseCategoricalCrossentropy()
    metrics=tf.keras.metrics.SparseCategoricalAccuracy()

    # Input Specific Parameters
    img_dim_x, img_dim_y, img_dim_z = 0, 0, 0

    # Dataset Specific Parameters
    amount_classes = 0
    if dataset=="mnist":
        img_dim_x, img_dim_y = 28, 28
        img_dim_z = 1    
        amount_classes = 10

    if dataset=="cifar10":
        img_dim_x, img_dim_y = 32, 32
        img_dim_z = 3
        amount_classes = 10

    if model_type == "cnn":
        model = keras.Sequential()

        model.add(layers.Conv2D(img_dim_x, (3,3), padding='same', activation='relu', input_shape=(img_dim_x,img_dim_y,img_dim_z)))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(img_dim_x, (3,3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(img_dim_x*2, (3,3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(img_dim_x*2, (3,3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Dropout(0.5))

        model.add(layers.Conv2D(img_dim_x*4, (3,3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(img_dim_x*4, (3,3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Dropout(0.5))

        model.add(layers.Flatten())
        model.add(layers.Dense(img_dim_x*4, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(amount_classes))
        model.add(layers.Activation('softmax'))
        
        model.compile(loss=loss_fn, optimizer=optimizer, metrics=[metrics])
        return model

    if model_type == "mlp":
        model = keras.Sequential()
        model.add(layers.Flatten(input_shape=(img_dim_x, img_dim_y, img_dim_z)))
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dense(amount_classes))
        model.add(layers.Activation("softmax"))
        model.compile(loss=loss_fn, optimizer=optimizer, metrics=[metrics])
        return model 


