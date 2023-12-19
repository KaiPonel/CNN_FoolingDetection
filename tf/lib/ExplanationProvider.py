import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten
from tf_explain.core import GradCAM, IntegratedGradients, VanillaGradients, GradientsInputs, OcclusionSensitivity
from tf_explain.callbacks.grad_cam import GradCAMCallback
from tf_explain.callbacks.vanilla_gradients import VanillaGradientsCallback
from tf_explain.callbacks.gradients_inputs import GradientsInputsCallback
from tf_explain.callbacks.occlusion_sensitivity import OcclusionSensitivityCallback

"""" Memory Introspection stuff """
#from memory_profiler import profile
#@profile
def run_explainers(x_in, y_in, model, NUM_IMAGES=300):
    print(len(x_in))
    """
        Runs all supported explainers on a certain model given sets of x and y and a limit of images
        `x_in`: np array of images 
        `y_in`: np array of labels
        `model`: trained model to be introspected
        `NUM_IMAGES`: Number of (x,y) pairs to be investigated. Defaults to 300.
    """
    explainer_grad = GradCAM()
    grids_gradcam = []
    grids_gradcam_guided = []
    explainer_ig = IntegratedGradients()
    grids_ig = []
    explainer_g_t_i = GradientsInputs()
    grids_g_t_i = []
    explainer_vgrads = VanillaGradients()
    grids_vgrads = []
    explainer_occlusion = OcclusionSensitivity()
    grids_occlusion = []

    # Iterate over the test_dataset for NUM_IMAGES. Run all the explanations using tf-explain and store the results in lists to be accessed later
    for index, (x_example, y_example) in enumerate(zip(x_in, y_in)):
        if index == NUM_IMAGES:
            break
        print("image: {}".format(index))
        x_example = ([x_example], None)
        y_example_index = int(y_example)

        grids_gradcam.append(explainer_grad.explain(x_example, model, class_index=y_example_index, use_guided_grads=False))
        grids_gradcam_guided.append(explainer_grad.explain(x_example, model, class_index=y_example_index, use_guided_grads=True))
        grids_ig.append(explainer_ig.explain(x_example, model, class_index=y_example_index))
        grids_g_t_i.append(explainer_g_t_i.explain(x_example, model, class_index=y_example_index))
        grids_vgrads.append(explainer_vgrads.explain(x_example, model, class_index=y_example_index))
        grids_occlusion.append(explainer_occlusion.explain(x_example, model, class_index=y_example_index, patch_size=7))
    
    return [(grids_gradcam, "gradcam"), 
            (grids_gradcam_guided, "gradcam_guided"), 
            (grids_ig, "ig"), 
            (grids_g_t_i, "g_t_i"), 
            (grids_vgrads, "vgrads"), 
            (grids_occlusion, "occlusion")]