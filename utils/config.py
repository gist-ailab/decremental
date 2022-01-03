import os
import yaml


def load_config(config_path):
  assert os.path.isfile(config_path), "No Configure file in {}".format(config_path)
  with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)
  return cfg  

