import torch
import torch.nn as nn
from ..consts import LEN_TRAINSET, LEN_TESTSET

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

        # self.len_embeddings = None (either LEN_TRAINSET or LEN_TESTSET) 
                                # len(train), batch_size, num_classes
        self.embeddings = np.zeros(16200, 16, 10)      
        # self.embeddings_initial_layer= np.zeros(self.len_embeddings, BATCH_SIZE, 64))
        # self.embeddings_down_block1 = np.zeros(self.len_embeddings, BATCH_SIZE/2, 32))
        # self.embeddings_down_block2 = np.zeros(self.len_embeddings, BATCH_SIZE/4, 16))
        # self.embeddings_bottleneck = np.zeros(self.len_embeddings, BATCH_SIZE, 64))
        # self.embeddings_fc = np.zeros(self.len_embeddings, BATCH_SIZE, 64)
        self.idx = 0

    def forward(self, x):
        # current_embeddings = dict()

        x1 = self.initial_layer(x)
        # current_embeddings['layer1'] = x1.detach().cpu()

        x2 = self.down_block1(x1)
        # current_embeddings['layer2'] = x2.detach().cpu()

        x3 = self.down_block2(x2)
        # current_embeddings['layer3'] = x3.detach().cpu()

        x_mid = self.bottleneck(x3)
        # current_embeddings['bottleneck'] = x_mid.detach().cpu()
        
        out = self.pool(x_mid)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        # current_embeddings['fc_out'] = out.detach().cpu()

        self.embeddings[self.idx] = out.detach().cpu().numpy()
        self.idx = self.idx + 1

        # self.embeddings.append(current_embeddings)

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
