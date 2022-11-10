from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Net(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        hidden_layers: int = 10,
        neurons_per_layer: int = 10,
        input_scaling: torch.Tensor = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float),
    ):
        super().__init__()
        self.input_scaling = input_scaling
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_dimension, neurons_per_layer))
        for _ in range(0, hidden_layers):
            layer = nn.Sequential(
                nn.PReLU(),
                nn.Linear(neurons_per_layer, neurons_per_layer),
            )
            self.hidden_layers.append(layer)
        self.hidden_layers.append(
            nn.Sequential(
                nn.PReLU(),
                nn.Linear(neurons_per_layer, output_dimension),
            )
        )

        self.apply(self._init_weights)

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            # module.weight.data.normal_(mean=0.0, std=1.0)
            torch.nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        x = self.input_scaling * x
        for layer in self.hidden_layers:
            x = layer(x)

        return x


def load_dataset(size: int = -1) -> TensorDataset:
    data = np.loadtxt("scatterdata.csv", delimiter=",")

    if size < 0 or size > data.shape[0]:
        x = data[:, :3]
        y = data[:, 3].reshape(-1, 1)
    else:
        x = data[:size, :3]
        y = data[:size, 3].reshape(-1, 1)

    return TensorDataset(
        torch.tensor(x, dtype=torch.float),
        torch.tensor(y, dtype=torch.float),
    )


def main() -> None:
    torch.manual_seed(42)

    batch_size = 50
    epochs = 50

    dataset = load_dataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    iscale = 1.0 / torch.std(dataset.tensors[0], dim=0)
    print(iscale)
    model = Net(
        input_dimension=dataset.tensors[0].shape[1],
        output_dimension=dataset.tensors[1].shape[1],
        hidden_layers=5,
        neurons_per_layer=30,
        input_scaling=iscale,
    )

    # Instantiate the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-3,
        weight_decay=5e-3,
    )

    # Instantiate the loss function
    loss_function = nn.MSELoss(reduction="mean")

    # The training loop
    for epoch in range(0, epochs):
        print(f"Starting epoch {epoch+1}")

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(loader, 0):
            # Get inputs
            inputs, targets = data

            # Set the gradients to zero
            optimizer.zero_grad()

            # Perform forward pass
            outputs = model(inputs)

            # Compute loss
            loss = loss_function(outputs, targets)

            # Perform backward pass
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

            # Perform optimization
            optimizer.step()

            # Print statistics
            current_loss += loss.item()

        print(current_loss)

    # Export the model as a torchscript file
    model.to(torch.device("cpu"))
    model.eval()
    sm = torch.jit.optimize_for_inference(torch.jit.script(model))
    sm.save("pretrained_model.pt")


if __name__ == "__main__":
    main()
