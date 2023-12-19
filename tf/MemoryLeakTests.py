import tensorflow as tf
import os
from lib.ModelProvider import get_model
from lib.DataProvider import load_MNIST, trim_lists
from lib.FeatureProvider import save_grids, load_model_weights
from lib.ExplanationProvider import run_explainers
from lib.AttackProvider import run_random_model_param_test_cascading

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model_mnist_cifar = get_model(model_type="cnn", dataset="mnist")
model_mnist_cifar = load_model_weights(model=model_mnist_cifar, model_name="cnn", dataset="mnist")
_, _, xTe, yTe = load_MNIST()
xTe, yTe = trim_lists(xTe, yTe, 20) # Only take 20 elements for now

run_random_model_param_test_cascading(model=model_mnist_cifar, x_test=xTe, y_test=yTe, model_name="cnn2", dataset="mnist")