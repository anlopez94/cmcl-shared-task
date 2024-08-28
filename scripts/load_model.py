import pathlib
folder_name=str(pathlib.Path(__file__).parent.resolve().parent.resolve())
import sys
sys.path.append('../')
sys.path.append(folder_name)
from src.model import RobertaRegressionModel
import torch

# Create a new instance of the model
  # Ensure the architecture matches the saved state
model_name='roberta-base'
device = torch.device('cuda')
model = RobertaRegressionModel(model_name).to(device)
# Load the saved state dictionary
model_path = '/model/model.pth'
model.load_state_dict(torch.load(folder_name + model_path))

model.eval()