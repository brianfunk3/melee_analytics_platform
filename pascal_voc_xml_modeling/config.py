import torch
# universal singular device creation to imported instead of passed around
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')