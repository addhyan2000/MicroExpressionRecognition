import torch
import torch.nn as nn

class AblatedSLSTTLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(16, 16)
        self.terminal_drop = nn.Dropout()
        self._init_specific_weights()
        
    def _init_specific_weights(self):
        print("Calling _init_specific_weights on", self.__class__.__name__)
        for name, param in self.lstm.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)

m = AblatedSLSTTLSTM()

print("One pass loop:")
for child in m.modules():
    print(child.__class__.__name__)
    if hasattr(child, 'reset_parameters'):
        print(" -> Calling reset_parameters")
    if hasattr(child, '_init_specific_weights'):
        print(" -> Calling _init_specific_weights")

