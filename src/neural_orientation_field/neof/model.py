import torch
import torch.nn as nn


class NeOFModel(nn.Module):
    def __init__(self):
        super(NeOFModel, self).__init__()


class NeOFCoarseModel(NeOFModel):
    def __init__(self, filter_size=128, num_encoding_functions=4):
        super(NeOFCoarseModel, self).__init__()

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Input layer (default: 42 -> 128)
        # 3 (position) + 3 (direction) + 3 * 2 * 6 (6 sin and cos encoding for position only)
        self.layer1 = nn.Linear(
            3 * 2 + 3 * 2 * num_encoding_functions,
            filter_size
        )
        self.layer1.apply(init_weights)
        # Layer 2 (default: 128 -> 128)
        self.layer2 = nn.Sequential(
            nn.Linear(filter_size, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, filter_size),
            nn.ReLU(),
        )
        self.layer2.apply(init_weights)
        # Layer 3 (default: 128 -> 4)
        self.layer3 = nn.Linear(filter_size, 7)
        self.layer3.apply(init_weights)
        # Short hand for nn.functional.relu
        self.relu = nn.functional.relu

    def forward(self, input: torch.Tensor):
        x = self.relu(self.layer1(input))
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class NeOFFineModel(NeOFModel):
    def __init__(self, filter_size=128, num_encoding_functions=8):
        super(NeOFFineModel, self).__init__()

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Input layer (default: 42 -> 128)
        # 3 (position) + 3 (direction) + 3 * 2 * 6 (6 sin and cos encoding for position only)
        self.layer1 = nn.Linear(
            3 * 2 + 3 * 2 * num_encoding_functions,
            filter_size
        )
        self.layer1.apply(init_weights)
        # Layer 2 (default: 128 -> 128)
        self.layer2 = nn.Sequential(
            nn.Linear(filter_size, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, filter_size),
            nn.ReLU(),
        )
        # Layer 3 (default: 128 + 42 -> 128)
        self.layer3 = nn.Sequential(
            nn.Linear(filter_size + 3 * 2 + 3 * 2 *
                      num_encoding_functions, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, filter_size),
            nn.ReLU(),
        )
        self.layer3.apply(init_weights)
        # Layer 4 (default: 128 -> 4)
        self.layer4 = nn.Linear(filter_size, 7)
        self.layer4.apply(init_weights)
        # Short hand for nn.functional.relu
        self.relu = nn.functional.relu

    def forward(self, input: torch.Tensor):
        x = self.relu(self.layer1(input))
        x = self.layer2(x)
        x = self.layer3(torch.concat((x, input), dim=-1))
        x = self.layer4(x)
        return x
