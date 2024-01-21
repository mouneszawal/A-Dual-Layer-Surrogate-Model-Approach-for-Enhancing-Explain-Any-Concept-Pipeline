from __future__ import print_function, division
import torch
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
from torchvision import transforms
import numpy as np
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import argparse

from sklearn import metrics
import os
from torchvision import models, transforms

import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from sklearn import metrics
import random
import warnings
warnings.filterwarnings("ignore")

from sam_explainer import *


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.008, help='lr')
parser.add_argument('--epoch', type=int, default=100, help='epoch')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--model', type=str, default='resnet50')
parser.add_argument('--data', type=str, default='imagenet')


args = parser.parse_args()
print('using dataset ',args.data)

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
fix_seed(args.seed)

print("using sam")
sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
# sam.to("cuda")
mask_generator = SamAutomaticMaskGenerator(sam)
    


class COCODataset(Dataset):
    def __init__(self, annotation_path, image_dir, transform=None):
        self.coco = COCO(annotation_path)
        self.image_dir = image_dir
        self.transform = transform
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        print(path)
        timage_dir = os.path.join(self.image_dir, path)
        # Load image
        image = Image.open(timage_dir).convert('RGB')

        # Apply transformations
        if self.transform is not None:
            image = self.transform(image)

        # You can modify this part to suit your needs
        # For simplicity, I am just using the first category from the annotations
        category_ids = [ann['category_id'] for ann in coco_annotation]
        category_id = category_ids[0] if category_ids else -1  # Use -1 or any identifier for 'no-category'

        return image, category_id, timage_dir

    def __len__(self):
        return len(self.ids)


test_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_reshape = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
])


image_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]) 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if args.model == 'resnet50':
    model = models.resnet50(weights='IMAGENET1K_V2').to(device)
    cvmodel = model.cuda()
    cvmodel.eval()
    feat_exp = create_feature_extractor(cvmodel, return_nodes=['avgpool'])
    fc = model.fc



model.eval()
feat_exp.eval()
cvmodel.eval()


        

auc_total = 0.0
auc_total_list = []

data_iter = None

# Set paths to the annotation file and image directory
annotation_path = 'D:/Downloads/coco2017/annotations/instances_val2017.json'  # Adjust path
image_dir = 'D:/Downloads/coco2017/val2017/'  # Adjust path

# Create the COCO Dataset
coco_dataset = COCODataset(annotation_path, image_dir, transform=test_preprocess)

# DataLoader
img_net = DataLoader(coco_dataset, batch_size=128)
# img_loader = DataLoader(coco_dataset,batch_size=128)
print("Total number of images in dataset:", len(img_net.dataset))

### random select 10000 imagenet sample to explain
num_sample = min(10000, len(img_net.dataset))
idx_path = "coco_random/{}_selected.pkl".format(num_sample)

if not os.path.isfile(idx_path):
    print("creating random list "+idx_path)
    data_iter = random.sample(list(range(len(img_net.dataset))),num_sample)
    with open(idx_path,'wb') as f:
        pickle.dump(data_iter, f)
else:
    print("loading random list "+idx_path)
    data_iter = pickle.load(open(idx_path,'rb'))

print("explaining {} images".format(num_sample))

num_sample = len(data_iter)

net_type = "Enhanced" # | Orig for original surrogate model - Enhanced for enhanced surrogate model
delete = False
test_type = "Deletion" if delete else "Insertion"

print(num_sample)
for idx in tqdm(data_iter):
    x , y, img_dir = coco_dataset[idx]
    clean_pil_load_img = Image.open(img_dir).convert('RGB')
    x = test_preprocess(clean_pil_load_img)
    with torch.no_grad():
        soft_org = torch.nn.functional.softmax(cvmodel(x.unsqueeze(0).cuda()),dim=1)
    image_class = int(torch.argmax(soft_org))
    probabilitie_org = float(torch.max(soft_org))
    for_mask_image = np.array(image_reshape(clean_pil_load_img)) ### np int type

    input_image_copy = for_mask_image.copy()
    concept_masks = None
    org_masks = gen_concept_masks(mask_generator,input_image_copy)
    concept_masks = np.array([i['segmentation'].tolist() for i in org_masks])
    auc_mask, shap_list = samshap(cvmodel,input_image_copy,image_class,concept_masks,fc,feat_exp,image_norm=image_norm,lr=args.lr,net_type=net_type)
    if type(auc_mask) == float:
        auc_total = auc_total + auc_mask
        continue

    if delete:
        auc_mask = 1 - auc_mask
        
    val_img_numpy = np.expand_dims(input_image_copy,0)
    val_img_numpy = (val_img_numpy * auc_mask).astype(np.uint8)
    batch_img = []
    for i in range(val_img_numpy.shape[0]): 
        batch_img.append(image_norm(val_img_numpy[i,:,:,:]))

    batch_img = torch.stack(batch_img).cuda()

    with torch.no_grad():
        out = torch.nn.functional.softmax(cvmodel(batch_img),dim=1)[:,image_class]
    out = out.cpu().numpy()

    out[out>= probabilitie_org] = probabilitie_org ### norm the upper bound of output to the original acc
    out = out/probabilitie_org
    x_axis = np.array(list(range(1,out.shape[0]+1)))/out.shape[0] *100
    if x_axis.shape[0] == 1:
        auc_tmp = float(out)
    else:
        auc_tmp = float(metrics.auc(x_axis, out))

    auc_total = auc_total + auc_tmp
    
print("this is the mean auc ",auc_total/num_sample)

