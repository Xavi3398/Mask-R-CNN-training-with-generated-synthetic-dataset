import os
import numpy as np
import skimage
from PIL import Image
import matplotlib.pyplot as plt
import time
import zipfile
import urllib.request
import shutil
import json
import shutil
from IPython.display import clear_output
import datetime
from enum import Enum

# COCO tools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from pycococreatortools import pycococreatortools

# Other .py
import utils
from config import Config
import model as modellib
import mail

# DIRECTORIES
ROOT_DIR = os.getcwd() # Root directory of the project
XAVI_DIR = os.path.join(os.getcwd(), "XAVI_Dataset") # Xavi Dataset directory
MODEL_DIR = os.path.join(XAVI_DIR, "model") # Directory to save trained model
DEFAULT_LOGS_DIR = os.path.join(MODEL_DIR, "logs") # Directory to save logs
MODEL_FILE = os.path.join(MODEL_DIR, "mask_rcnn_xavi.h5") # Local path to trained weights file
DEFAULT_TRAINING_SUBSET = "train512"
DEFAULT_VALIDATION_SUBSET = "val512"




class MasksType(Enum):
    ON_THE_FLY = 1
    PNG = 2
    ANNOTATIONS = 3




class XaviConfig(Config):
    NAME = "xavi512"
    BACKBONE = "resnet101"

    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 5000
    TRAIN_ROIS_PER_IMAGE = 200
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    DETECTION_MIN_CONFIDENCE = 0.8 # Skip detections with < 80% confidence
    
    def __init__(self, json_categories):    
        XaviConfig.NUM_CLASSES = 1 + len(json_categories) # Background + json classes
        super(XaviConfig, self).__init__()





class XaviDataset(utils.Dataset):

# --------------------- Load Dataset ---------------------------
    def load_xavi(self, dataset_dir, subset, type_masks=MasksType.PNG, json_categories = None, size_image = None, class_ids = None):

        self.main_dir = dataset_dir
        self.subset = subset
        self.dataset_dir = os.path.join(dataset_dir, subset)
        self.type_masks = type_masks
        
        # Subdirs
        self.rgb_dir = os.path.join(self.dataset_dir, "RGB")
        self.cat_dir = os.path.join(self.dataset_dir, "Categories")
        self.seg_dir = os.path.join(self.dataset_dir, "Segmentation")
        self.msk_dir = os.path.join(self.dataset_dir, "Masks")
        
        if type_masks == MasksType.PNG or type_masks == MasksType.ON_THE_FLY:
            assert(json_categories != None and size_image != None)
            self.load_categories(json_categories, size_image)
        elif type_masks == MasksType.ANNOTATIONS:
            self.load_annotations(class_ids)

    # Load classes from categories.json
    def load_categories(self, json_categories, size_image):
        self.json_categories = json_categories
        for i, cat in enumerate(self.json_categories):
            self.add_class("xavi", i + 1, cat['name'])
        
        # Add rgb images from RGB folder
        self.size_image = size_image
        width, height = size_image
        for i, filename in enumerate(os.listdir(self.rgb_dir)):
            image_path = os.path.join(self.rgb_dir, filename)

            self.add_image(
                "xavi",
                image_id = i + 1,
                path = image_path,
                width = width, 
                height = height)

    # Load classes from annotations.json
    # Select subset of classes
    def load_annotations(self, class_ids):
        # Load json of annotations
        coco = COCO(os.path.join(self.dataset_dir, "annotations.json"))

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            class_ids = sorted(coco.getCatIds())
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("xavi", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "xavi", image_id=i,
                path=os.path.join(self.rgb_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))





# --------------------- Get Masks ---------------------------

    def load_mask(self, image_id):
        
        if self.type_masks == MasksType.ON_THE_FLY:
            return self.load_mask_on_the_fly(image_id)
        elif self.type_masks == MasksType.PNG:
            return self.load_mask_png(image_id)
        elif self.type_masks == MasksType.ANNOTATIONS:
            return self.load_mask_annotations(image_id)
        else:
            raise Exception("Invalid value for type of masks:" + str(self.type_masks))

    def load_mask_on_the_fly(self, image_id):
        masks, class_ids = self.generate_mask(image_id)
        if masks.size == 0:
            return super(XaviDataset, self).load_mask(image_id)
        else:
            return masks, class_ids
            
    def load_mask_png(self, image_id):
        image_dir = os.path.join(self.msk_dir, os.path.splitext(os.path.basename(self.image_info[image_id]['path']))[0])
        class_ids = []
        masks = []

        for i, filename in enumerate(os.listdir(image_dir)):

            # Mask
            mask = np.array(Image.open(os.path.join(image_dir, filename)), dtype=np.uint8) // 255
            masks.append(mask)

            # Class
            image_class = filename[0:filename.rfind('_')]
            class_ids.append(self.class_names.index(image_class))
        
        if len(masks) == 0:
            return super(XaviDataset, self).load_mask(image_id)
        else:
            return np.stack(masks, axis=2), np.array(class_ids, dtype=np.int32)

    def load_mask_annotations(self, image_id):
        image_info = self.image_info[image_id]
        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]

        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "xavi.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(XaviDataset, self).load_mask(image_id)

    # Compute BW individual masks from Category and Segmentation images
    def generate_mask(self, image_id):

        def only_color(color, image):
            r = color[0]
            g = color[1]
            b = color[2]
            RESULT = np.where(image[:,:,0] == r, 1, 0) * np.where(image[:,:,1] == g, 1, 0) * np.where(image[:,:,2] == b, 1, 0)
            return np.array(RESULT, dtype=np.uint8)
        
        image_name = os.path.splitext(os.path.basename(self.image_info[image_id]['path']))[0] + ".png"
        CAT_IMAGE = np.array(Image.open(os.path.join(self.cat_dir, image_name)), dtype=np.uint8)
        SEG_IMAGE = np.array(Image.open(os.path.join(self.seg_dir, image_name)), dtype=np.uint8)
        categories = self.json_categories
        
        class_ids = []
        masks = []
        
        i = 1
        for cat in categories:

            # Get category mask
            MASK_CATEGORY = only_color(cat['color'], CAT_IMAGE)

            # Leave only one category in segmentation image
            SEG_IMAGE2 = np.copy(SEG_IMAGE)
            SEG_IMAGE2[:,:,0] *= MASK_CATEGORY
            SEG_IMAGE2[:,:,1] *= MASK_CATEGORY
            SEG_IMAGE2[:,:,2] *= MASK_CATEGORY

            # Compute colors left
            diff_colors = np.unique(SEG_IMAGE2.reshape(-1, SEG_IMAGE2.shape[2]), axis=0)

            for color in diff_colors:

                # if color is not black
                if color[0] != 0 or color[1] != 0 or color[2] != 0:
                    MASK_SEGMENTATION = only_color(color, SEG_IMAGE2)
                    masks.append(MASK_SEGMENTATION)
                    
                    # Since classes are loaded from the same json, they are in the same order
                    class_ids.append(i)
            i += 1
                    
        return np.stack(masks, axis=2), np.array(class_ids, dtype=np.int32)


    # Convert annotation which can be polygons, uncompressed RLE to RLE
    # From pycocotools with a few changes.
    def annToRLE(self, ann, height, width):
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    # Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask
    # From pycocotools with a few changes.
    def annToMask(self, ann, height, width):
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

# --------------------- Clean Dataset ---------------------------

    # Removes images with no classes from the dataset
    def remove_black_images(self, target_path):

        # Find black images
        black_images = self.get_no_class()

        # Subfolders
        t_rgb_folder = os.path.join(target_path, "RGB")
        t_cat_folder = os.path.join(target_path, "Categories")
        t_seg_folder = os.path.join(target_path, "Segmentation")

        # Create folders
        if not os.path.exists(t_rgb_folder):
            os.makedirs(t_rgb_folder)
        if not os.path.exists(t_cat_folder):
            os.makedirs(t_cat_folder)
        if not os.path.exists(t_seg_folder):
            os.makedirs(t_seg_folder)

        # Move black images
        for image in black_images:
            shutil.move(os.path.join(self.rgb_dir, image + ".jpg"), os.path.join(t_rgb_folder, image + ".jpg"))
            shutil.move(os.path.join(self.cat_dir, image + ".png"), os.path.join(t_cat_folder, image + ".png"))
            shutil.move(os.path.join(self.seg_dir, image + ".png"), os.path.join(t_seg_folder, image + ".png"))

        print("Number of Images removed from dataset: " + str(len(black_images)))

    # Returns list of images with no classes
    def get_no_class(self):

        no_class = []
        for i, filename in enumerate(os.listdir(self.cat_dir)):
            image = Image.open(os.path.join(self.cat_dir, filename))
            if not image.getbbox():
                no_class.append(os.path.splitext(os.path.basename(filename))[0])

        return no_class


    # Generate Masks as PNGs
    def GeneratePngMasks(self):

        # Create folder
        if not os.path.exists(self.msk_dir):
            os.makedirs(self.msk_dir)
        
        start = time.time()

        # for each image
        for i, image_id in enumerate(self.image_ids):

            image_masks, image_classes = self.generate_mask(image_id)
            image_dir = os.path.join(self.msk_dir, os.path.splitext(os.path.basename(self.image_info[image_id]['path']))[0])

            # Create folder
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)

            # for each mask
            for j, image_class in enumerate(image_classes): 
                Image.fromarray(image_masks[:,:,j] * 255).save(os.path.join(image_dir, self.class_names[image_class] + "_" + str(j) + ".png"), "PNG")

            if(i % 100 == 0):
                print("Count: {:7} / {:7} \t || Time from start: {:7}".format(i, self.num_images, str(round( time.time() - start, 2))))

        clear_output(wait=True)
        print("Finished generating masks. Time elapsed: " + str(round( time.time() - start, 2)))

    # Convert PNG generated masks to annotations.json
    def PngMasksToAnnotations(self, description = "Xavi Dataset"):

        coco_output = {}

        # info
        coco_output['info'] = {
            "description": description,
            "url": "",
            "version": "1.0",
            "year": 2021,
            "contributor": "Xavi",
            "date_created": datetime.datetime.utcnow().isoformat(' ')
        }

        # licenses
        coco_output['licenses'] = []
        coco_output['licenses'].append({
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        })

        # categories
        coco_output['categories'] = []
        class_ids = {}
        for i, category in enumerate(self.json_categories):
            coco_output['categories'].append({
                'id': (i + 1),
                'name': category['name'],
            })
            class_ids[category['name']] = (i + 1)

        # images and annotations
        rgb_images = os.listdir(self.rgb_dir)
        n_images = len(rgb_images)
        segmentation_id = 1
        start = time.time()
        coco_output["images"] = []
        coco_output["annotations"] = []

        # go through each RGB image
        for i, rgb_image in enumerate(rgb_images):
            
            # Image info
            image_id = i + 1
            image_info = pycococreatortools.create_image_info(
                image_id, # id
                os.path.basename(rgb_image), # name
                self.size_image # size
            )
            coco_output["images"].append(image_info)

            # go through associated png masks
            folder = os.path.join(self.msk_dir, os.path.splitext(os.path.basename(rgb_image))[0])
            for mask_image in os.listdir(folder):
                
                class_name = mask_image[0:mask_image.rfind('_')]
                class_id = class_ids[class_name]

                category_info = {'id': class_id, 'is_crowd': 0}
                binary_mask = np.asarray(Image.open(os.path.join(folder, mask_image)).convert('1')).astype(np.uint8)
                annotation_info = pycococreatortools.create_annotation_info(
                    segmentation_id, 
                    image_id, 
                    category_info, 
                    binary_mask,
                    self.size_image, # size 
                    tolerance=2)
                if annotation_info is not None:
                    coco_output["annotations"].append(annotation_info)
                segmentation_id += 1
            
            if(i % 100 == 0):
                print("Count: {:7} / {:7} \t || Time from start: {:7}".format(i, n_images, str(round( time.time() - start, 2))))

        # Save as json
        json.dump(coco_output, open(os.path.join(self.dataset_dir, "annotations.json"), 'w'))

        clear_output(wait=True)
        print("Finished generating annotations.")
        print("Saved to: " + os.path.join(self.dataset_dir, "annotations.json"))
        print("Elapsed time: " + str(round( time.time() - start, 2)))




# --------------------- Other ---------------------------

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "xavi":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
    
    def plot_masks(self, masks, class_ids):
        fig=plt.figure(figsize=(50, 50))
        Y_size = class_ids.size // 4
        if class_ids.size % 4 != 0:
            Y_size += 1
        for i in range(class_ids.size):
            s = fig.add_subplot(Y_size, 4, i + 1)
            s.set_title(class_ids[i], fontsize=32)
            plt.imshow(Image.fromarray(masks[:,:,i]))

# Get subset of annotations
def AnnotationsSubset(coco_input_path, list_ids, name, path, list_classes = [], change_classes = []):

    coco_input = COCO(coco_input_path)
    coco_output = {}
    start = time.time()

    # info
    coco_output['info'] = coco_input.dataset['info']
    coco_output['info']['description'] = coco_output['info']['description'] + "(subset: " + name + ")"
    coco_output['info']["date_created"] = datetime.datetime.utcnow().isoformat(' ')

    # licenses
    coco_output['licenses'] = coco_input.dataset['licenses']

    # categories
    coco_output['categories'] = coco_input.loadCats(list_classes)

    # change category ids
    for (a, b) in change_classes:
        for cat in coco_output['categories']:
            if cat["id"] == a:
                cat["id"] = b
    coco_output['categories'].sort(key= lambda e: e["id"])

    # images and annotations
    coco_output["images"] = coco_input.loadImgs(list_ids)
    coco_output["annotations"] = coco_input.loadAnns(coco_input.getAnnIds(imgIds=list_ids, catIds=list_classes))
    
    # change category ids for annotations
    for (a, b) in change_classes:
        for annotation in coco_output["annotations"]:
            if annotation["category_id"] == a:
                annotation["category_id"] = b

    # Save as json
    json.dump(coco_output, open(os.path.join(path, name + ".json"), 'w'))

    clear_output(wait=True)
    print("Finished generating annotations.")
    print("Saved to: " + os.path.join(path, name + ".json"))
    print("Elapsed time: " + str(round( time.time() - start, 2)))


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on Generated Dataset.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' on Generated Dataset")
    parser.add_argument('--dataset', required=False,
                        default=XAVI_DIR,
                        metavar="/path/to/GeneratedDataset/",
                        help='Directory of the Generated Dataset')
    parser.add_argument('--train_subset', required=False,
                        default=DEFAULT_TRAINING_SUBSET,
                        metavar="<train_subset>",
                        help='Subset to use (default train512)')
    parser.add_argument('--val_subset', required=False,
                        default=DEFAULT_VALIDATION_SUBSET,
                        metavar="<val_subset>",
                        help='Subset to use (default val512)')
    parser.add_argument('--model', required=False,
                        default="",
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file ('xavi', 'last', 'imagenet), default=None")
    parser.add_argument('--masksType', required=False,
                        default="annotations",
                        metavar="<masksType>",
                        help="Type of mask to use ('on_the_fly', 'png', 'annotations), default=annotations")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=dataset/model/logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--mail', required=False,
                        default=False,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    args = parser.parse_args()
    print("\n\nCommand: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Training subset: ", args.train_subset)
    print("Validation subset: ", args.val_subset)
    print("Logs: ", args.logs)
    print("Limit: ", args.limit)
    print("Mailing notifications: ", args.mail)

    # Configurations
    json_categories = json.load(open(os.path.join(XAVI_DIR, args.train_subset, "categories.json")))["Categories"]
    config = XaviConfig(json_categories)
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)

    # Select weights file to load
    load_weights = True
    if args.model.lower() == "xavi":
        model_path = MODEL_FILE
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()[1]
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    elif args.model != "":
        model_path = args.model
    else:
        load_weights = False

    # Load weights
    if(load_weights):
        print("Loading weights ", model_path)
        model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":

        total_ep = 0

        # Training dataset.
        dataset_train = XaviDataset()
        dataset_train.load_xavi(args.dataset, args.train_subset, MasksType[args.masksType.upper()], json_categories, (512, 512))
        dataset_train.prepare()
        print("\n----------------- Training dataset loaded -------------------\n")

        # Validation dataset
        dataset_val = XaviDataset()
        dataset_val.load_xavi(args.dataset, args.val_subset, MasksType[args.masksType.upper()], json_categories, (512, 512))
        dataset_val.prepare()
        print("\n----------------- Validation dataset loaded -------------------\n")

        message = "\n----------------- Start of training -------------------\n"
        print(message)
        if args.mail:
            mail.send_text(message)

        # Training - Stage 1
        # Train all layers
        print("\n\Train all layers")
        it = 16
        ep = 2
        for i in range(it):
            total_ep += ep
            model.train(dataset_train, dataset_val,
                        learning_rate=config.LEARNING_RATE,
                        epochs=total_ep,
                        layers='all')

            # Feedback
            message = "\n\n----------------- End of Stage 1 (all layers)- " + str(i + 1) + "/" + str(it) + " -------------------\n\n"
            print(message)
            if args.mail:
                mail.send_text(message)

        # Training - Stage 2
        # Fine tune all layers
        print("\n\nFine tune all layers")
        it = 4
        ep = 2
        for i in range(it):
            total_ep += ep
            model.train(dataset_train, dataset_val,
                        learning_rate=config.LEARNING_RATE / 10,
                        epochs=total_ep,
                        layers='all')

            # Feedback
            message = "\n\n----------------- End of Stage 2 (Finetune all layers)- " + str(i + 1) + "/" + str(it) + " -------------------\n\n"
            print(message)
            if args.mail:
                mail.send_text(message)

        message = "\n\n----------------- End of Training -------------------\n\n"
        print(message)
        if args.mail:
            mail.send_text(message)

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
    
