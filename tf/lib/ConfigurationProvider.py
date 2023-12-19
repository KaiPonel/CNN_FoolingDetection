import tensorflow as tf

def get_EarlyStopping():
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,  
        restore_best_weights=True  
    )