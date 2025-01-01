import torch
import torch.nn as nn

from shortcut_connections import ExampleDeepNeuralNetwork


layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1.0, 0.0, 1.0]])
torch.manual_seed(123)

model_without_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes=layer_sizes, use_shortcut=False
)


def print_gradients(model, x):

    output = model(x)
    target = torch.tensor([[0.0]])

    loss = nn.MSELoss()
    loss = loss(output, target)
    loss.backward()

    for name, param in model.named_parameters():
        if "weight" in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


print_gradients(model_without_shortcut, sample_input)

torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(
    layer_sizes=layer_sizes, use_shortcut=True
)

print_gradients(model_with_shortcut, sample_input)
