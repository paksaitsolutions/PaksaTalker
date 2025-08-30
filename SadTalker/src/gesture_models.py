import torch
import torch.nn as nn

class GRUGAN(nn.Module):
    def __init__(self, device):
        super(GRUGAN, self).__init__()
        self.device = device
        self.gru = nn.GRU(input_size=80*16, hidden_size=256, num_layers=2, batch_first=True)
        self.linear = nn.Linear(256, 70)
        
    def forward(self, audio_features):
        # Process audio features through GRU
        output, _ = self.gru(audio_features)
        # Get the last time step's output
        gesture_output = self.linear(output[:, -1, :])
        return gesture_output
