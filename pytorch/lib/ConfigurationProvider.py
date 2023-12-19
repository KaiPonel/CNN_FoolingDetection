from pytorch_lightning.callbacks import EarlyStopping


def get_EarlyStopping():
    return EarlyStopping(monitor='val_loss', patience=3)