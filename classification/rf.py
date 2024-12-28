import torch
from torch import nn
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from timm.utils import AverageMeter
from torch.nn import functional as F

def compute_erf(model, layer_name, input_size=(224, 224), device='cpu'):
    """
    Compute the Effective Receptive Field (ERF) for a specified layer in a model.

    Parameters:
        model (nn.Module): The neural network model.
        layer_name (str): The name of the layer to analyze.
        input_size (tuple): Size of the input image (height, width).
        device (str): Device to perform computations on ('cpu' or 'cuda').

    Returns:
        np.ndarray: The computed ERF as a 2D numpy array.
    """
    model.to(device)
    model.eval()

    # Select the layer for ERF visualization
    layer = dict([*model.named_modules()])[layer_name]

    # Register a forward hook to capture the feature map
    activation = {}
    def hook_fn(module, input, output):
        activation['feature_map'] = output

    hook = layer.register_forward_hook(hook_fn)

    erfs = AverageMeter()

    for _ in range(100):
        # Create a dummy input image with a single activated pixel
        input_image = torch.randn(1, 3, *input_size, device=device)
        input_image = input_image - input_image.min()
        input_image.requires_grad_()

        # Forward pass through the model
        _ = model(input_image)

        # Retrieve the feature map from the selected layer
        feature_map = activation['feature_map']

        # Compute the gradient of the output with respect to the input
        output = F.relu(feature_map[0, :, feature_map.size(2) // 2, feature_map.size(3) // 2]).sum()
        model.zero_grad()
        output.backward()

        # Get the gradient at the input
        gradient = input_image.grad[0].cpu().numpy().sum(axis=0)

        # Compute the ERF by normalizing the gradient
        erf = gradient
        erf[erf < 0] = 0
        erfs.update(erf)

    erf = erfs.avg
    erf /= erf.max()

    # Clean up
    hook.remove()

    return erf

if __name__ == '__main__':
    # Load a pretrained ResNet model
    from models.vrwkv7 import VRWKV7

    model = VRWKV7(**{'img_size': 224, 'patch_size': 4, 'in_channels': 3, 'drop_path_rate': 0.2, 'embed_dims': 96, 'num_heads': 4, 'num_classes': 1000, 'depth': [2, 2, 8, 2], 'norm_layer': 'ln2d', 'key_norm': True, 'final_norm': True, 'dims': 96})
    model.load_state_dict(torch.load("tmp/vrwkv7_tiny_0230s/20241226164556/ckpt_epoch_64.pth")['model'])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_named_params = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
    for name, param in model.named_parameters():
        print(f'{name = }, {param.numel() = }')
    print(f"{n_params = }, {n_named_params = }")
    exit(0)

    model = models.resnet50()

    # Compute the ERF for a specific layer
    layer_name = 'layer3'  # Example: 'layer3' in ResNet-50
    erf = compute_erf(model, layer_name, input_size=(224, 224), device='cuda' if torch.cuda.is_available() else 'cpu')

    # Plot the ERF
    plt.imshow(erf, cmap='hot', interpolation='nearest')
    plt.colorbar(label='ERF Intensity')
    plt.title(f'Effective Receptive Field of {layer_name} in ResNet50')
    plt.savefig("images/rf.jpg")
    plt.show()
