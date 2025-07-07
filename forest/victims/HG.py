import torch
import torch.nn as nn

class HG(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super(HG, self).__init__()
        
        # --- Encoder Path ---
        self.initial_layer = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), # Added Batch Norm
            nn.ReLU()
        )
        
        self.down_block1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2) # 64x64 -> 32x32
        )
        
        self.down_block2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2) # 32x32 -> 16x16
        )

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # --- Decoder Path ---
        # Note: We are discarding the decoder for classification.
        # If this were for segmentation, we would build the up-sampling path here.

        # --- Classifier Head ---
        # This replaces the entire decoder and the flawed final layers.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(256, 128), # From bottleneck's 256 channels
            nn.ReLU(),
            nn.Dropout(0.5), # Add dropout for regularization
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Encoder Path
        x1 = self.initial_layer(x)
        x2 = self.down_block1(x1)
        x3 = self.down_block2(x2)
        
        # Bottleneck
        x_mid = self.bottleneck(x3)
        
        # Classification Head
        out = self.pool(x_mid)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out

# --- Example of usage ---
if __name__ == "__main__":
    # Create a dummy input tensor (e.g., batch_size=4, 3 channels, 64x64 image)
    dummy_input = torch.randn(4, 3, 64, 64)
    
    # Instantiate the model for 10 classes
    model = HG_Corrected(num_classes=10)
    
    # Get the output
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Model Architecture:\n{model}")
    print(f"Output shape: {output.shape}") # Should be [4, 10]
    
    assert output.shape == (4, 10)
    print("\nModel forward pass successful!")
