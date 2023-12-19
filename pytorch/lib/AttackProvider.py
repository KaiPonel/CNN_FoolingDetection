import torch
import random

from .ModelExecutionProvider import ModelExecutionProvider
from .FeatureProvider import save_grids, load_model_weights
from .ExplanationProvider import run_explainers
from .ModelProvider import get_model
from .DataProvider import load_dataset, convert_to_td_and_dl


def run_all_attacks_on_all_models_on_all_datasets(num_images = 1000, models=["cnn", "mlp"], datasets=["cifar10", "mnist"], base_dir="/project/FoolingDetection/pytorch/grids"):
    for model in models:
        for dataset in datasets:
            # 1. No Attack
            model_correct = load_model_weights(model_type=model, dataset=dataset)
            grids = run_explainers(dataset, model_correct, NUM_IMAGES=num_images)
            save_grids(grids_dict=grids, base_dir=base_dir, dataset=dataset, model_type=model, attack="correct", override=True)
            # 2. Random Labels
            model_random_labels = train_random_label_model(model_name=model, dataset=dataset)
            grids = run_explainers(dataset, model_random_labels, NUM_IMAGES=num_images)
            save_grids(grids_dict=grids, base_dir=base_dir, dataset=dataset, model_type=model, attack="RandLabels", override=True)
            # 3A. Random Params Sequential
            #run_random_model_param_test_sequential(model=model_correct, model_type=model, dataset=dataset, num_images=num_images, base_dir=base_dir)
            # 3B. Random Params Cascading
            #run_random_model_param_test_cascading(model=model_correct, model_type=model, dataset=dataset, num_images=num_images, base_dir=base_dir)


def run_all_attacks_on_all_models_on_all_datasets2(num_images = 1000, models=["mlp"], datasets=["cifar10", "mnist"]):
    for model in models:
        for dataset in datasets:
            # 1. No Attack
            model_correct = load_model_weights(model_type=model, dataset=dataset)
            grids = run_explainers(dataset, model_correct, NUM_IMAGES=num_images)
            save_grids(grids_dict=grids, base_dir="/project/FoolingDetection/pytorch/grids", dataset=dataset, model_type=model, attack="correct", override=True)
            # 2. Random Labels
            model_random_labels = train_random_label_model(model_name=model, dataset=dataset)
            grids = run_explainers(dataset, model_random_labels, NUM_IMAGES=num_images)
            save_grids(grids_dict=grids, base_dir="/project/FoolingDetection/pytorch/grids", dataset=dataset, model_type=model, attack="RandLabels", override=True)
            # 3A. Random Params Sequential
            run_random_model_param_test_sequential(model=model_correct, model_type=model, dataset=dataset, num_images=num_images)
            # 3B. Random Params Cascading
            run_random_model_param_test_cascading(model=model_correct, model_type=model, dataset=dataset, num_images=num_images)

    # for all attacks
        # for all models
            # for all datasets
                
                # run attack

                # save attack results
    


                


    # Much code
    return 



def run_random_model_param_test_cascading(model, model_type="cnn", dataset="mnist", num_images=20, base_dir="/project/FoolingDetection/pytorch/grids"):
    """
        Runs the explainers on a model with randomized parameters. (Cascading) \n
        `model`: trained model to run the explainers on \n
        `model_type`: name of the model to be trained \n
        `dataset`: dataset the model should be trained on \n
        `num_images`: number of images to run the explainers on \n
    """

    # Get all layers in the Sequential block
    layers = list(model.children())[0]
    
    # Iterate over the layers of the model in reverse order (from last to first)
    for i, layer in enumerate(reversed(list(layers))):
        # Print the layer name and wether the layer has weights
        print("Layer at index {}: {} - Has weights: {}".format(i, layer.__class__.__name__, hasattr(layer, 'weight')))
        if hasattr(layer, 'weight'):
            randomized_weights = torch.rand_like(layer.weight)
            layer.weight.data = randomized_weights
            if hasattr(layer, 'bias') and layer.bias is not None:
                randomized_bias = torch.rand_like(layer.bias)
                layer.bias.data = randomized_bias
            # Get the name of the layer
            layer_name = layer.__class__.__name__
            # Run the explainers on the randomized model
            grids = run_explainers(dataset, model, NUM_IMAGES=num_images)
            # Save the grids

            save_grids(grids_dict=grids, base_dir=base_dir, dataset=dataset, model_type=model_type, attack="RandParams/Cascading/{}_{}".format(i, layer_name), override=True)

def run_random_model_param_test_sequential(model, model_type="cnn", dataset="mnist", num_images=20, base_dir="/project/FoolingDetection/pytorch/grids"):
    """
        Runs the explainers on a model with randomized parameters. (Sequential) \n
        `model`: trained model to run the explainers on \n
        `model_type`: name of the model to be trained \n
        `dataset`: dataset the model should be trained on \n
        `num_images`: number of images to run the explainers on \n
    """

    # Get all layers in the Sequential block
    layers = list(model.children())[0]
    
    # Iterate over the layers of the model in reverse order (from last to first)
    for i, layer in enumerate(reversed(list(layers))):
        # Print the layer name and wether the layer has weights
        print("Layer at index {}: {} - Has weights: {}".format(i, layer.__class__.__name__, hasattr(layer, 'weight')))
        if hasattr(layer, 'weight'):
            randomized_weights = torch.rand_like(layer.weight)
            # Store the actual weights of the layer so they can be restored later
            actual_weights = layer.weight.data

            layer.weight.data = randomized_weights
            if hasattr(layer, 'bias') and layer.bias is not None:
                randomized_bias = torch.rand_like(layer.bias)
                layer.bias.data = randomized_bias
            # Get the name of the layer
            layer_name = layer.__class__.__name__
            # Run the explainers on the randomized model
            grids = run_explainers(dataset, model, NUM_IMAGES=num_images)
            # Save the grids
            save_grids(grids_dict=grids, base_dir=base_dir, dataset=dataset, model_type=model_type, attack="RandParams/Sequential/{}_{}".format(i, layer_name), override=True)
            # Restore the actual weights of the layer
            layer.weight.data = actual_weights




def train_random_label_model(model_name="cnn", dataset="mnist"):
    """
        Trains a model on random labels. \n
        `x_train`: training data \n
        `y_train`: training labels \n
        `x_test`: test data \n
        `y_test`: test labels \n
        `model_name`: name of the model to be trained \n
        `dataset`: dataset the model should be trained on \n

        Returns the trained model on random labels.
    """

    xTrain, yTrain, xTest, yTest = load_dataset(name=dataset)

    # Generate random labels
    def generate_random_labels(y):
        unique_values = list(set(y))
        return [random.choice(unique_values) for _ in y]

    # Generate random labels for training data
    yTrain_random = torch.tensor(generate_random_labels(yTrain))
    # Get the model architecture (untrained)
    random_label_model_untrained = get_model(model_type=model_name, dataset=dataset)

    # Define the data loaders
    trainloader = convert_to_td_and_dl(xTrain, yTrain_random, batch_size=128)
    valloader = convert_to_td_and_dl(xTest, yTest, batch_size=128)

    # Define the model execution provider and train the model on the random labels
    executor = ModelExecutionProvider(model=random_label_model_untrained)
    trained_random_label_model = executor.train(train_dl=trainloader, test_dl=valloader, n_epochs=10)

    # Return the trained model on random labels
    return trained_random_label_model

