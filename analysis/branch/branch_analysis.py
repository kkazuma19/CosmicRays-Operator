from s_deeponet import SequentialDeepONet
from s_mionet import SequentialMIONet
import torch
import torch.nn as nn

def init_single_model():
    ''' Initialize the model architecture '''
    dim = 128
    model = SequentialDeepONet(
        branch_type='lstm',
        branch_input_size=12,
        branch_hidden_size=128,
        branch_num_layers=4,
        branch_output_size=dim,
        trunk_architecture=[2, 128, 128, dim],
        num_outputs=1,
        use_transform=False,
        activation_fn=nn.ReLU,
    )
    return model

def init_multi_model():
    dim = 128
    # Define a single branch configuration to be reused
    base_branch_config = {
        "type": "lstm",
        "input_size": 1,
        "hidden_size": 128,
        "num_layers": 4,
        "output_size": dim
    }

    # Create a dictionary with the same branch configuration for 12 branches
    branches_config = {f"sensor{i+1}": base_branch_config for i in range(12)}

    # Trunk network configuration
    trunk_architecture = [2, 128, 128, dim]
    num_outputs = 1

    # Instantiate the model with the replicated branches
    model = SequentialMIONet(branches_config, trunk_architecture, num_outputs ,use_transform=False)

    return model

def load_model_experiment(type, model_path):
    ''' Load model from a given path '''
    if type == "single":
        model = init_single_model()
    elif type == "multi":
        model = init_multi_model()
    else:
        raise ValueError(f"Unknown model type: {type}")

    model.load_state_dict(torch.load(model_path))
    
    # check if correctly loaded
    if model is None:
        raise ValueError(f"Failed to load model from {model_path}")
    print(f"Loaded model from {model_path}")
    
    model.eval()
    return model