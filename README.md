# Training of Mask R-CNN with a generated synthetic dataset
This repository contains all needed code to train Mask R-CNN with a custom generated dataset. To see the used implementation of Mask R-CNN, please, visit https://github.com/matterport/Mask_RCNN.

All code to load the dataset, the configuration and the training can be found in the [xavi.py](xavi.py) file. Note that some changes have been made to the files of the original project ([utils.py](utils.py), [visualize.py](visualize.py) and [model.py](model.py)) in order to add some features and update some others.

## Jupyter notebooks
* **[XAVI - 1 - Dataset Cleansing.ipynb](<XAVI - 1 - Dataset Cleansing.ipynb>)**: takes a synthetic dataset and removes all useless images.
* **[XAVI - 2 - Masks generation (PNG).ipynb](<XAVI - 2 - Masks generation (PNG).ipynb>)**: generates instance segmentation masks for the synthetic dataset using other types of images created by the generator as PNG images stored in folders.
* **[XAVI - 3 - Masks generation (COCO annotations).ipynb](<XAVI - 3 - Masks generation (COCO annotations).ipynb>)**: generates instance segmentation masks for the synthetic dataset using other types of images created by the generator as an annotations file in COCO format.
* **[XAVI - 4 - Masks loading speed comparison.ipynb](<XAVI - 4 - Masks loading speed comparison.ipynb>)**: compares load speed of PNG masks, annotations masks, and generation of masks at runtime.
* **[XAVI - 4 - Masks loading speed comparison.ipynb](<XAVI - 4 - Masks loading speed comparison.ipynb>)**: compares load speed of PNG masks, annotations masks, and generation of masks at runtime.
* **[XAVI - 5 - Dataset Info.ipynb](<XAVI - 5 - Dataset Info.ipynb>)**: counts number of images containig each class and total instances of each class in a dataset. This method is very slow and methods of the coco api should be used to get this information from the annotations file instead.
* **[XAVI - 6 - Training evaluation.ipynb](<XAVI - 6 - Training evaluation.ipynb>)**: computes precission for the weights after each epoch with both synthetic and COCO images. Models after each epoch are not available due to their total size.
* **[XAVI - Subset creation for retraining.ipynb](<XAVI - Subset creation for retraining.ipynb>)**: tool for subset creation from a COCO format annotations file.
* **[Masks Generator](<Masks Generator/Masks Generator.ipynb>)**: example of the followed processed to generate the labelled instance masks for the synthetic dataset.
* **[XAVI - demo.ipynb](<XAVI - demo.ipynb>)**: demonstration of the tools of the project taken from the original repo, but with the synthetic dataset.
* **[XAVI - inspect_data.ipynb](<XAVI - inspect_data.ipynb>)**: notebook to inspect the synthetic dataset and try the visualization tools, taken from the original repo, but changed to work with the synthetic dataset.
* **[XAVI - inspect_model.ipynb](<XAVI - inspect_model.ipynb>)**: test to load and use the custom model, also taken from the original repo and modified to work with the custom trained model and synthetic dataset.

## Requirements of the project
* Version of used packages and libraries can be found in the requirements.txt file.
* Custom synthetic dataset and trained model can be found [here](https://drive.google.com/drive/folders/1MLFq8opsd18-M4HBoUHNCJF5_-YwivWh?usp=sharing).
* [MS COCO 2014 val dataset](https://cocodataset.org/) with the 5k [minival](https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0) and the 35K [validation-minus-minival](https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0) subsets to run evaluation.

## Other Files of interest
Look [here](https://drive.google.com/drive/folders/1MLFq8opsd18-M4HBoUHNCJF5_-YwivWh?usp=sharing) for all related files of the project for the dataset generation with Unity. Since this work is part of my final project for the Degree, the user manual and the report are only available in spanish, because of lack of time for its translation, even though all other files (including jupyter notebooks) and comments are in english. These are all the resources made available:

* **"XAVI Dataset" folder**: Synthetic dataset for training a neural net in image segmentation of instances. There are two versions of the dataset available, each of them with training, validation and testing subsets: complete version (large and with lots of files) and reduced version (only RGB images and annotations in COCO format).
* **"Image Generator.rar"**: Program used to create the dataset, along with all the configuration files.
* **"Image Generator with Unity.rar"**: Unity project used create the "Image Generator" program. Contains all tools needed for automating the production of synthetic datasets.
* **"model"**: obtained weights at the last epoch of training (40) and log files of each epoch, which can be plotted using TensorBoard. 
* **"Generación procedural para entrenamiento de redes neuronales.pdf"**: Memory of the final project of the degree, regarding the production of synthetic datasets with Unity and the training of a sample neural net.
* **"TFG - Manual de usuario.pdf"**: User manual of the Unity project and the developed tools for datasets generation.
* **"Resulting images"**: results obtained with the model trained with synthetic data and a model trained with COCO images both with generated images and with COCO images.

## Example images
### Synthetic dataset image examples
<p float="left">
  <img src="/sample_images/examples/photo512_000000000628_City.jpg" width="300" /> 
  <img src="/sample_images/examples/photo512_000000009914_Forest.jpg" width="300" />
  <img src="/sample_images/examples/photo512_000000012248_Park.jpg" width="300" /> 
</p>

<p float="left">
  <img src="/sample_images/examples/photo512_000000012335_Home.jpg" width="300" />
  <img src="/sample_images/examples/photo512_000000015964_LuxuriousHome.jpg" width="300" /> 
  <img src="/sample_images/examples/photo512_000000019041_Musuem.jpg" width="300" />
</p>

### Synthetic dataset image examples with their info images
The info images consist of a semantic image with humans in red and cars in blue, and an instance segmentation image, where all objects are in a different color. Using both, instance segmentation masks labelled with the name of the class can be obtained as shown in [Masks Generator.ipynb](<Masks Generator/Masks Generator.ipynb>).
<p float="left">
  <img src="/sample_images/examples_with_masks/photo512_000000002682_City.jpg" width="300" /> 
  <img src="/sample_images/examples_with_masks/photo512_000000002682_City.png" width="300" />
  <img src="/sample_images/examples_with_masks/photo512_000000002682_City_2.png" width="300" />
</p>

<p float="left">
  <img src="/sample_images/examples_with_masks/photo512_000000013915_Home.jpg" width="300" /> 
  <img src="/sample_images/examples_with_masks/photo512_000000013915_Home.png" width="300" />
  <img src="/sample_images/examples_with_masks/photo512_000000013915_Home_2.png" width="300" />
</p>

### Best results of the trained model with synthetic images
Each prediction consists of a detection box, a segmentation mask and the IOU score.
<p float="left">
  <img src="/sample_images/XAVI-XAVI/XAVI XAVI - 1.000 - photo512_000000000344_City.png" width="500" /> 
  <img src="/sample_images/XAVI-XAVI/XAVI XAVI - 1.000 - photo512_000000008497_LuxuriousHome.png" width="500" />
</p>

### Best results of the trained model with COCO images
Each prediction consists of a detection box, a segmentation mask and the IOU score.

<p float="left">
  <img src="/sample_images/XAVI-COCO/XAVI COCO 1.000 - COCO_val2014_000000054931.png" width="500" /> 
  <img src="/sample_images/XAVI-COCO/XAVI COCO 1.000 - COCO_val2014_000000512985.png" width="500" />
</p>
