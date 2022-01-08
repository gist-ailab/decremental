import torch
import os
from ptflops import get_model_complexity_info
from utils import *
from torchsummary import summary

data_format = {
  "cifar": (3, 32, 32),
  "imagenet": (3, 32, 32),
}  


def model_summary(model, data='cifar'):
  summary(model, data_format[data])



def calculate_flops(model, data="cifar"):
  return get_model_complexity_info(model, data_format[data], as_strings=True,
                                  print_per_layer_stat=True, verbose=True)



