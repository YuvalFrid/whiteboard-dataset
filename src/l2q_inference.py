#!/usr/bin/env python3
# l2q_inference.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -------------------------
# Model definition
# -------------------------
class L2QGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)   # 28->14
        self.enc2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)  # 14->7
        self.enc3 = nn.Conv2d(64, 128, 3, stride=2, padding=1) # 7->4

        # Bottleneck
        self.fc1 = nn.Linear(128*4*4, 256)
        self.fc2 = nn.Linear(256, 128*4*4)

        # Decoder
        self.dec1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=0)  # 4->7
        self.dec2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, output_padding=0)   # 7->14
        self.dec3 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1, output_padding=0)    # 14->28

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(x.size(0), 128, 4, 4)

        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = torch.sigmoid(self.dec3(x))
        return x

# -------------------------
# Inference helper
# -------------------------
class L2QInference:
    def __init__(self, weights_path="src/l2q_generator_weights.pth", device=None):
        self.device = device or torch.device("cpu")#da" if torch.cuda.is_available() else "cpu")
        self.model = L2QGenerator().to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def predict(self, l_image):
        """
        l_image: 28x28 numpy array (values in [0,1])
        Returns: 28x28 numpy array prediction of '?'
        """
        x_tensor = torch.from_numpy(l_image.astype("float32")).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(x_tensor)
        return pred.squeeze().cpu().numpy()

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Load your l image here as numpy array, shape (28,28)
    # Example placeholder: random binary image
    l_example = np.random.randint(0,2,(28,28)).astype(np.float32)

    infer = L2QInference(weights_path="src/l2q_generator_weights.pth")
    q_pred = infer.predict(l_example)

    # Visualize
    plt.subplot(1,2,1)
    plt.imshow(l_example, cmap="gray")
    plt.title("Input l")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(q_pred, cmap="gray")
    plt.title("Predicted ?")
    plt.axis("off")
    plt.show()

