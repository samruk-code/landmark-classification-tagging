import torch
import torch.nn as nn

class ResidualBlock(nn.Module):

    def __init__(self, in_channel, out_channel, stride):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch2 = nn.BatchNorm2d(out_channel)

        self.skip_connection = nn.Sequential()
        if stride!=1 or in_channel!= out_channel:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channel)
            )
            
    def forward(self, x):
        output = self.relu(self.batch1(self.conv1(x)))
        output = self.batch2(self.conv2(output))
        output += self.skip_connection(x)
        return self.relu(output)

# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        self.res_block1 = ResidualBlock(3, 64, 1)
        self.res_block2 = ResidualBlock(64, 128, 2)
        self.res_block3 = ResidualBlock(128, 256, 2)
        self.res_block4 = ResidualBlock(256, 512, 2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(192, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.global_avg_pool(x)
        x = self.head(x)
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
