"""
 * Copyright (C) 2019 Zhonghui You
 * If you are using this code in your research, please cite the paper:
 * Gate Decorator: Global Filter Pruning Method for Accelerating Deep Convolutional Neural Networks, in NeurIPS 2019.

*** TransTailer Algorithm ***

  Input : Pre-trained Model, Source Data, Target Data
  Output: Optimal Sub-Model

"""
import timm



'''1. Load Model'''
pretrained_model = timm.create_model('resnet50', pretrained=True)
source_data = None
target_data = None


'''2. Initialize scaling factor'''



'''2.1 Convert BatchNorm2d -> GatedBatchNorm2d'''

'''3. Initialize FC layer to target data'''

optimal_model = None


'''4. TransTailer Pruning'''

while True:
  #1. Train scaling factor
  #2. Transfrom scaling factor to global importance
  #3. Pruning based on global importance
  #4. Fine-tune for target data
  #5. Check Accuracy Drop
  
  pass
  