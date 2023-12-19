import numpy as np
import random

from .FeatureProvider import save_grids
from .ExplanationProvider import run_explainers
from .ModelProvider import get_model
from .ConfigurationProvider import get_EarlyStopping

"""" Memory Introspection stuff """
#from memory_profiler import profile
#@profile
def run_random_model_param_test_cascading(model, x_test, y_test, model_name="cnn", dataset="mnist"):
    # Loop over layers: Randomize one layer at a time
    for i in range(len(model.layers)-1, -1, -1):
        # Randomize the weights of the current layer if it has weights
        layer = model.layers[i]
        weights = layer.get_weights()
        if weights:  # Check if the layer has weights
            randomized_weights = [np.random.rand(*w.shape) for w in weights]
            layer.set_weights(randomized_weights)
            layer_name = layer.name
            grids = run_explainers(x_test, y_test, model, len(x_test))
            save_grids(grids_list=grids, base_dir="/project/FoolingDetection/grids", dataset=dataset, model=model_name, attack="RandParams/Cascading/{}_{}".format(i, layer_name), override=True)


def train_random_label_model(x_train, y_train, x_test, y_test, model_name="cnn", dataset="mnist"):
    def generate_random_labels(y):
        unique_values = list(set(y))
        return [random.choice(unique_values) for _ in y]
    
    y_train_random = np.asarray(generate_random_labels(y_train))
    random_label_model = get_model(model_type=model_name, dataset=dataset)
    print(x_train.shape)
    print(y_train.shape)
    random_label_model.fit(x_train, y_train_random, epochs=100, batch_size=128, validation_data=(x_test, y_test), callbacks=[get_EarlyStopping()])
    return random_label_model

    