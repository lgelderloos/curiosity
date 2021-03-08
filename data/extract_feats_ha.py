import numpy as np
import torch
import torchvision
import PIL
import os
import pickle
import json
from bs4 import BeautifulSoup

# helper function to get python tuple of ints instead of numpy array of unsigned
# small integers
def as_int_tuple(np_box):
    box = [int(i) for i in np_box]
    return tuple(box)

def get_coordinates(bnbox):
    # takes the xml coordinates and returns a tuple (ymin, xmin, ymax, xmax)
    ymin = int(bnbox.find('ymin').string)
    xmin = int(bnbox.find('xmin').string)
    ymax = int(bnbox.find('ymax').string)
    xmax = int(bnbox.find('xmax').string)
    return (ymin, xmin, ymax, xmax)

def get_bnbox(soup):
    objs = soup.find_all('object')
    golds = {}
    for obj in objs:
        names = obj.find_all('name')
        bndboxes = obj.find_all('bndbox')
        if len(bndboxes) > 0:
            boxes = [get_coordinates(box) for box in bndboxes]
            for name in names:
                if name.string in golds:
                    golds[name.string].extend(boxes)
                else:
                    golds[name.string] = boxes
    return golds

torch.device('cuda')

###############################################################################
# definitions for normalize & reshape
normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
resize = torchvision.transforms.Resize((224, 224))

###############################################################################

# get model
vgg16 = torchvision.models.vgg16(pretrained=True)
# getting prefinal rather than final layer
new_classifier = torch.nn.Sequential(*list(vgg16.classifier.children())[:-1])
vgg16.classifier = new_classifier
for param in vgg16.parameters():
          print(type(param.data), param.size())
#set model to eval mode to disable dropout
vgg16.eval()
# to cuda
vgg16.to("cuda")

torchvision.set_image_backend('PIL')

###############################################################################
# Global variable: list of non-processable images
exceptional_images = []
small = []
imgdir = "F30k/flickr30k_images/"

# make list of all the img ids
fnames = os.listdir(imgdir)
imglist = [f.split(".")[0] for f in fnames]
# dictionary to keep track of all the start&stop points for each obj in each img
objects_to_feats = {}

###############################################################################
# for each image in the dataset
for img in imglist:
    # load the image
    try:
        image = PIL.Image.open((imgdir+img+".jpg"))
        # convert to RGB if necessary
        image = image.convert(mode="RGB")
    except:
        # skip image if it cannot be loaded & converted to rgb
        exceptional_images.append(img)
        continue

    # load associated soup
    with open(("F30k/Flickr30kEntities/Annotations/"+img+".xml"), "r") as f:
        contents = f.read()
    soup = BeautifulSoup(contents, "xml")
    # extract object bboxes
    box_by_id = get_bnbox(soup)

    objects_to_feats[img] = {} # links obj ids to the corresponding npy array dims
    feats = [] # temporary storage for feats (will be concatenated)
    # for each bbox in this image:
    nth_box = 0
    for id in box_by_id:
        start_feat = nth_box # keep track of starting point
        for bbox in box_by_id[id]:
            skipbox = False
            # select that part of the image. note the bbox coordinates & subtraction
            torch_coords = (bbox[0], # upper pixel coordinate (ymin)
                        bbox[1], # left pixel coordinate (xmin)
                        (bbox[2]-bbox[0]), # height: ymax - ymin
                        (bbox[3]-bbox[1])) # width: xmax - xmin
            # checking for negatives/zero-dimensional axes
            for coo in torch_coords:
                if not coo >= 1:
                    print("for {} this box {} led to these torch coords {}. skipping this box.".format(img, bbox, torch_coords))
                    skipbox = True
            if skipbox:
                continue
            crop = torchvision.transforms.functional.crop(image, *torch_coords)
            # reshaping crop to fit vgg inputsize
            crop = resize(crop)
            # to tensor
            crop = torchvision.transforms.ToTensor()(crop).cuda()
            # normalize it
            crop = normalize(crop)
            # get visual_features
            cropfeats = vgg16.forward(crop.unsqueeze(0)).squeeze()
            # appending the .data of cropfeats to feats
            feats.append(cropfeats.data)
            nth_box += 1
        # log which feats belong to this object (which may contain multiple bboxes)
        # don't log objects for which there is no non-emtpy box
        if not start_feat == nth_box:
            objects_to_feats[img][id] = (start_feat, nth_box)
        else:
            print("no good bbox for object {} in img {}".format(id, img), flush = True)
    if len(feats) > 0:
        img_feats = torch.stack(feats)
        # write feats to disk
        torch.save(img_feats, "ha_bbox_vggs/{}.pt".format(img))
    else:
        print("no features for this image: {}".format(img), flush = True)

try:
    with open("ha_vgg_indices.json", 'w') as f:
        json.dump(objects_to_feats, f)
    print("\nSuccesfully saved mapping of objects to vgg feature indices")
except:
    print("\nFailed to save mapping of objects to vgg feature indices")

print("\n\nDone\n")

print("\nThese images could not be processed:")
[print(i, end="\t") for i in exceptional_images]
print("\n{} imgs not processed".format(len(exceptional_images)))

if len(exceptional_images) > 0:
    try:
        with open("unprocessed.p", 'wb') as f:
            pickle.dump(exceptional_images, f)
    except:
        print("failed to save list of unprocessable imgs")
