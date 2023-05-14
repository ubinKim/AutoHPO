import torch
import torch.nn as nn

class TISv1(nn.Module):
    def __init__(self, dropout, hidden_unit):
        super(TISv1, self).__init__()
        self.features = \
            nn.Sequential(\
                # 1
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(7, 4)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 1)),
                # 2
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 1)),
                # 3
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 1))
                )

        self.classifier = \
            nn.Sequential(
                nn.Linear(in_features=64, out_features=hidden_unit),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(in_features=hidden_unit, out_features=2))

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Test run
if __name__ == "__main__":
    test_input = torch.randn((1, 1, 60, 4))
    model = TISv1(dropout=0.5, hidden_unit=32)
    out = model(test_input)
    print(out.shape)
