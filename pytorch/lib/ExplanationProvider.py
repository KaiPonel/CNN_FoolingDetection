import torch
import warnings

# https://captum.ai/docs/attribution_algorithms
from captum.attr import (
    IntegratedGradients,
    Occlusion,
    Saliency,
    GuidedBackprop,
    GuidedGradCam,
    LayerGradCam,
    DeepLift,
    ShapleyValues,
    Lime,
    InputXGradient,
    DeepLiftShap,
    ShapleyValueSampling,
)

from .DataProvider import load_dataset

def run_explainers(dataset, model, NUM_IMAGES=300):
    warnings.filterwarnings("ignore", category=UserWarning)

    # Create a dictionary of explainers
    explainers = {
        "ig": IntegratedGradients(model),
        "g_t_i": InputXGradient(model),
        "vgrads": Saliency(model),
        "dl": DeepLift(model),
        "backprop": GuidedBackprop(model),
        "lime": Lime(model),
        "shapeley_sampling": ShapleyValueSampling(model),
        "deep_lift_shap": DeepLiftShap(model)
    }

    # Load the dataset
    _, _, x_in, y_in = load_dataset(name=dataset, as_dict=False)

    if(dataset == "mnist"):
        image_shape = (1, 28, 28)  # (channels, height, width)
    elif(dataset == "cifar10"):
        image_shape = (3, 32, 32)  # (channels, height, width)

    # Create tensors for black, gray, and white images
    black_image = torch.zeros(image_shape, device='cuda:0')  # All pixel values are 0.0 (black)
    gray_image = torch.ones(image_shape, device='cuda:0') * 0.5  # All pixel values are 0.5 (gray)
    white_image = torch.ones(image_shape, device='cuda:0')  # All pixel values are 1.0 (white)

    # Stack these tensors along a new dimension to create a tensor with three examples
    baseline = torch.stack([black_image, gray_image, white_image])

    # Create a dictionary of grids
    grids = {key: [] for key in explainers.keys()}

    for index, (x_example, y_example) in enumerate(zip(x_in, y_in)):
        if index == NUM_IMAGES:
            break

        if (index+1) % 10 == 0:
            print(f"Running explainers on image {index+1} out of {NUM_IMAGES} ({(index+1)/NUM_IMAGES*100:.2f}%)")

        #debug:
        # print(x_example)

        x_example = torch.from_numpy(x_example).unsqueeze(0).to("cuda:0")
        model.to("cuda:0")

        y_example_index = int(y_example)

        for key in explainers.keys():
            if key == 'deep_lift_shap':
                attr = explainers[key].attribute(x_example, target=y_example_index, baselines=baseline)
            else:
                attr = explainers[key].attribute(x_example, target=y_example_index)

            attr = attr.cpu()
            grids[key].append(attr.squeeze().detach().numpy())

    # Normalize the grids between 0 and 1
    for key in grids.keys():
        grids[key] = [(grid - grid.min()) / (grid.max() - grid.min()) for grid in grids[key]]

    return grids
            
            