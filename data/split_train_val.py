import os
from glob import glob
from tqdm import tqdm
import numpy as np


def split_stanford_cars(data_dir):
    image_list = glob(os.path.join(data_dir, 'train_ori', '*.jpg'))
    train_folder = os.path.join(data_dir, 'train')
    val_folder = os.path.join(data_dir, 'val')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    
    train_list = np.random.choice(image_list, int(len(image_list) * 0.8), replace=False).tolist()
    val_list = list(set(image_list) - set(train_list))
    
    # Copy File
    for image_path in tqdm(train_list):
        old_path = image_path
        new_path = image_path.replace('train_ori', 'train')
        script = 'cp -r %s %s' %(old_path, new_path)
        os.system(script)
        
    for image_path in tqdm(val_list):
        old_path = image_path
        new_path = image_path.replace('train_ori', 'val')
        script = 'cp -r %s %s' %(old_path, new_path)
        os.system(script)
    

def split_food101(data_dir):
    # Image Folder
    image_folder = os.path.join(data_dir, 'images')
    train_folder = os.path.join(data_dir, 'train')
    val_folder = os.path.join(data_dir, 'val')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    
    # Image List
    with open(os.path.join(data_dir, 'meta/train.txt'), 'r') as f:
        train_list = f.readlines()
    
    with open(os.path.join(data_dir, 'meta/test.txt'), 'r') as f:
        val_list = f.readlines()
    
    # Copy File
    for image_path in tqdm(train_list):
        old_path = os.path.join(image_folder, image_path.strip() + '.jpg')
        new_path = os.path.join(train_folder, image_path.strip() + '.jpg')
        os.makedirs(new_path, exist_ok=True)

        script = 'cp -r %s %s' %(old_path, new_path)
        os.system(script)
        
    for image_path in tqdm(val_list):
        old_path = os.path.join(image_folder, image_path.strip() + '.jpg')
        new_path = os.path.join(val_folder, image_path.strip() + '.jpg')
        os.makedirs(new_path, exist_ok=True)
    
        script = 'cp -r %s %s' %(old_path, new_path)
        os.system(script)
    
def split_uec256(data_dir):
    # Image Folder
    image_folder = os.path.join(data_dir, 'images')
    train_folder = os.path.join(data_dir, 'train')
    val_folder = os.path.join(data_dir, 'val')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    
    # Image List
    image_list = glob(os.path.join(image_folder, '*', '*.jpg'))
    category_list = [image.split('/')[-2] for image in image_list]
    category_list = np.unique(category_list).tolist()
    
    train_list, val_list = [], []
    for category in category_list:
        image_list_ix = glob(os.path.join(image_folder, category, '*.jpg'))
        
        train_ix = np.random.choice(image_list_ix, int(len(image_list_ix) * 0.8), replace=False).tolist()
        val_ix = list(set(image_list_ix) - set(train_ix))
        
        train_list += train_ix
        val_list += val_ix
    
    # Copy File
    for image_path in tqdm(train_list):
        old_path = image_path
        new_path = image_path.replace('images', 'train')
        os.makedirs(os.path.dirname(new_path), exist_ok=True)

        script = 'cp -r %s %s' %(old_path, new_path)
        os.system(script)
        
    # Copy File
    for image_path in tqdm(val_list):
        old_path = image_path
        new_path = image_path.replace('images', 'val')
        os.makedirs(os.path.dirname(new_path), exist_ok=True)

        script = 'cp -r %s %s' %(old_path, new_path)
        os.system(script)


if __name__=='__main__':
    data_dir = '/data/sung/dataset/eurosat'
    split_uec256(data_dir)