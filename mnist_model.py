"""
MNIST CNN Model - Nhận dạng chữ số viết tay (0-9)

Architecture: CNN với 2 conv layers + 2 FC layers
Training: Tự động train trên MNIST dataset lần đầu chạy
Accuracy: ~99% trên MNIST test set
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image

# Path lưu model đã train
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'mnist_cnn.pth')


class MNISTNet(nn.Module):
    """
    CNN cho nhận dạng chữ số viết tay.

    Architecture:
        Input (1x28x28)
        → Conv2d(1→32, 3x3) + ReLU + Conv2d(32→64, 3x3) + ReLU + MaxPool(2x2) + Dropout(0.25)
        → Conv2d(64→128, 3x3) + ReLU + MaxPool(2x2) + Dropout(0.25)
        → Flatten → FC(128*5*5 → 256) + ReLU + Dropout(0.5)
        → FC(256 → 10)
    """

    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        self.dropout3 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # Block 1: Conv → Conv → Pool
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        # Block 2: Conv → Pool
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        # Flatten → FC
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


class DigitRecognizer:
    """
    Wrapper class cho MNIST CNN model.
    Tu dong train model lan dau neu chua co file saved.
    """

    def __init__(self, device=None):
        """
        Khoi tao DigitRecognizer.

        Args:
            device: 'cuda' hoac 'cpu'. Mac dinh tu dong detect.
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = MNISTNet().to(self.device)

        # Transform cho input image
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Load model
        if os.path.exists(MODEL_PATH):
            print("[MNIST] Loading saved model...")
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device, weights_only=True))
            self.model.eval()
            print("[MNIST] Model loaded successfully!")
        else:
            raise FileNotFoundError(f"[MNIST] Model file not found at {MODEL_PATH}. Vui long chay 'python train_model.py' de huan luyen mo hinh truoc.")

    def predict_digit(self, image):
        """
        Nhan dang 1 chu so tu anh.

        Args:
            image: PIL Image hoac numpy array (anh cua 1 chu so)

        Returns:
            dict: {digit, confidence, probabilities}
        """
        self.model.eval()

        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Ensure image is PIL
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be PIL Image or numpy array")

        # Preprocess
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.exp(output).cpu().numpy()[0]

        digit = int(np.argmax(probabilities))
        confidence = float(probabilities[digit])

        return {
            'digit': digit,
            'confidence': confidence,
            'probabilities': probabilities.tolist()
        }

    def predict_digits_batch(self, images):
        """
        Nhan dang nhieu chu so cung luc.

        Args:
            images: List cua PIL Image hoac numpy array

        Returns:
            List cua dict {digit, confidence}
        """
        self.model.eval()

        tensors = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            tensors.append(self.transform(img))

        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            output = self.model(batch)
            probabilities = torch.exp(output).cpu().numpy()

        results = []
        for i, probs in enumerate(probabilities):
            digit = int(np.argmax(probs))
            confidence = float(probs[digit])
            results.append({
                'digit': digit,
                'confidence': confidence
            })

        return results
