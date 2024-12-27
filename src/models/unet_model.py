import torch.nn as nn
from models.unet_model_dropout import UNet
from huggingface_hub import PyTorchModelHubMixin

class DeepFaultBaselineModel(
    nn.Module,
    PyTorchModelHubMixin, 
    # optionally, you can add metadata which gets pushed to the model card
    repo_url="porestar/deepfault-unet-baseline-full-augment",
    pipeline_tag="image-to-image",
    license="mit",
):
    def __init__(self):
        super().__init__()
        self.model = UNet(num_classes=2,
                          enc_channels=[1, 64, 128, 256, 512, 1024],
                          dec_channels=[1024, 512, 256, 128, 64],
                          dropout=0.0)

    def forward(self, x):
        x = self.model(x)
        return x
