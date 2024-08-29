import sys
import torch
import pandas as pd
import argparse
import pathlib
folder_name=str(pathlib.Path(__file__).parent.resolve().parent.resolve())
sys.path.append('../')
sys.path.append(folder_name)
import src.model


parser = argparse.ArgumentParser()

parser.add_argument('--num-ensembles', type=int, default=10)
parser.add_argument('--use-provo', type=bool, default=True)

# dev: use our own train/valid split
# submission: use all data and make predictions on unknown data
parser.add_argument('--mode', type=str, default='submission')

args = parser.parse_args()

if args.mode == 'dev':
  train_df = pd.read_csv(folder_name + "/data/training_data/train.csv")
  valid_df = pd.read_csv(folder_name + "/data/training_data/valid.csv")
else:
  train_df = pd.read_csv(folder_name + "/data/training_data/train_and_valid.csv")
  valid_df = pd.read_csv(folder_name + "/data/training_data/test_data.csv")

provo_df = pd.read_csv(folder_name + "/data/provo.csv")

for ensemble_ix in range(args.num_ensembles):
  model_trainer = src.model.ModelTrainer(model_name='roberta-base')

  if args.use_provo:
    model_trainer.train(provo_df, num_epochs=100)

  if args.mode == 'dev':
    model_trainer.train(train_df, valid_df, num_epochs=150)
  else:
    model_trainer.train(train_df, num_epochs=120)
  torch.save(model_trainer.model, folder_name + '/models/model_' + str(ensemble_ix) + '.pth')
  predict_df = model_trainer.predict(valid_df)
  predict_df.to_csv(f"scripts/predict-{ensemble_ix}.csv", index=False)
