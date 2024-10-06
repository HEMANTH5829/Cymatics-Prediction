import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import librosa

# Function to generate spectrogram from sound wave
def generate_spectrogram(wave, sample_rate=44100, n_fft=2048, hop_length=512):
    spectrogram = librosa.feature.melspectrogram(y=wave, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram

# Convolutional Neural Network for predicting patterns based on spectrogram
class CymaticPredictor(nn.Module):
    def __init__(self, input_channels=1, input_height=128, input_width=128):
        super(CymaticPredictor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate the size of the feature maps after convolutions and pooling
        self.feature_size = self._get_conv_output((input_channels, input_height, input_width))
        
        self.fc1 = nn.Linear(self.feature_size, 1024)
        self.fc2 = nn.Linear(1024, 100 * 100)

    def _get_conv_output(self, shape):
        bs = 1
        input = torch.rand(bs, *shape)
        output = self._forward_features(input)
        n_size = output.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, 100, 100)

# Function to augment data by adding noise
def augment_wave(wave):
    noise = np.random.normal(0, 0.01, wave.shape)  # Add Gaussian noise
    return wave + noise

# Generate a dataset of cymatic patterns for various sound waves
def generate_dataset(num_samples=100):
    input_spectrograms = []
    target_patterns = []
    
    for _ in range(num_samples):
        duration = 1  # seconds
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        freq = np.random.uniform(20, 2000)  # Random frequency between 20 Hz and 2 kHz
        
        # Generate original wave and augmented wave
        original_wave = np.sin(2 * np.pi * freq * t)
        augmented_wave = augment_wave(original_wave)

        # Generate spectrograms for both original and augmented waves
        spectrogram_original = generate_spectrogram(original_wave)
        spectrogram_augmented = generate_spectrogram(augmented_wave)

        # Ensure consistent dimensions
        spectrogram_original = librosa.util.fix_length(spectrogram_original, size=128, axis=0)
        spectrogram_original = librosa.util.fix_length(spectrogram_original, size=128, axis=1)
        
        input_spectrograms.append(spectrogram_original)

        pattern = np.random.rand(100, 100)  # Random 100x100 pattern
        target_patterns.append(pattern)

    input_spectrograms = torch.tensor(np.array(input_spectrograms), dtype=torch.float32).unsqueeze(1)
    target_patterns = torch.tensor(np.array(target_patterns), dtype=torch.float32)
    
    print(f"Input spectrograms shape: {input_spectrograms.shape}")
    print(f"Target patterns shape: {target_patterns.shape}")
    
    return input_spectrograms, target_patterns

# Main execution
if __name__ == "__main__":
    # Generate training data
    input_spectrograms, target_patterns = generate_dataset(num_samples=500)

    # Instantiate the model
    model = CymaticPredictor(input_channels=1, input_height=128, input_width=128)

    # Print model summary
    print(model)

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training loop
    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        output_patterns = model(input_spectrograms)
        loss = criterion(output_patterns, target_patterns)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    # Predicting a new pattern based on an unseen sound wave
    new_sound_wave = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))  # 440 Hz sine wave
    new_spectrogram = generate_spectrogram(new_sound_wave)
    new_spectrogram = librosa.util.fix_length(new_spectrogram, size=128, axis=0)
    new_spectrogram = librosa.util.fix_length(new_spectrogram, size=128, axis=1)
    new_spectrogram = torch.tensor(new_spectrogram, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        predicted_pattern = model(new_spectrogram).numpy()

    # Visualize the predicted cymatic pattern
    plt.figure(figsize=(8, 8))
    plt.imshow(predicted_pattern[0], cmap='viridis')
    plt.colorbar()
    plt.title('Predicted Cymatic Pattern for New Sound Wave')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()
