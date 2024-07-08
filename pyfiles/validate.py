import dataset
# #loading the json file
# import json
# # Load the JSON file
# with open('dict_indices.json', 'r') as file:
#     data = json.load(file)
# train_indices = data['train_indices']
# val_indices = data['val_indices']

# # Load the JSON file
# with open('domainb_indices.json', 'r') as file:
#     data = json.load(file)

# domainb_train = data['train']
# domainb_val = data['val']

import json

import albumentations as A
from albumentations.pytorch import ToTensorV2
import re


from torch.utils.data import ConcatDataset


# Load the JSON file
with open('dict_indices.json', 'r') as file:
    data = json.load(file)
train_indices = data['train_indices']
val_indices = data['val_indices']

with open('domainb_indices.json', 'r') as file:
    data = json.load(file)

domainb_indices = data['train']


train_path = os.path.join(os.getcwd(),"train")
mask_path = os.path.join(os.getcwd(),"train_gt")
train_x = os.listdir('train')
train_x.sort(key=extract_numerical_part)
train_y = os.listdir('train_gt')
train_y.sort(key=extract_numerical_part)

val_path = os.path.join(os.getcwd(),"val")
val_gt = os.path.join(os.getcwd(),"val_gt")
val_x = os.listdir('val')
val_x.sort(key=extract_numerical_part)
val_y = os.listdir('val_gt')
val_y.sort(key=extract_numerical_part)

testB = os.path.join(os.getcwd(),"imgs")
test_gtB = os.path.join(os.getcwd(),"Gts")
path_soft_gtB  =  os.path.join(os.getcwd(),"soft_gtB")

testB_images = os.listdir(testB)
testB_images.sort(key=extract_numerical_part)
testB_masks = os.listdir(test_gtB)
testB_masks.sort(key=extract_numerical_part)



vss = dess400(train_path,mask_path,train_indices,train_transform)
vall = dess400(val_path,val_gt,val_indices,valid_transform)
dbb = dess400(testB,test_gtB,domainb_indices,train_transform)
domainb_soft = dess400(testB , path_soft_gtB, domainb_indices, train_transform)
# softb = dess400(testB,path_soft_gtB,domainb_train , train_transform)
# testb_val = dess400(testB, test_gtB ,domainb_val,train_transform)

# combined_ds_train = ConcatDataset([vss,softb ])


# train_loader = DataLoader(vss, batch_size=5, shuffle=True)
# val_loader = DataLoader(vall, batch_size=15, shuffle=False)

# mix_loader = DataLoader(combined_ds_train, batch_size=10 , shuffle=True)

vs= Dess2(train_path,mask_path,train_x ,train_y , train_transform)

val= Dess2(val_path , val_gt, val_x,val_y , valid_transform)
domainb = hotencoded(testB , test_gtB , testB_images    , testB_masks , num_classes=7,transform=valid_transform)

# ds = hot400(train_path , mask_path,train_indices,num_classes=7,transform=train_transform)
# ds_val = hot400(val_path , val_gt , val_indices,num_classes=7,transform=valid_transform)
ds = hotencoded(train_path,mask_path , train_x,train_y , num_classes=7,transform=train_transform)
ds_val = hotencoded(val_path,val_gt,val_x,val_y,num_classes=7,transform=valid_transform)
