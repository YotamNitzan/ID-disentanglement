# Setup

## Environment

It's designed to use Tensorflow 2.X on python (3.7), using cuda 10.1 and cudnn 7.6.5.
Run `conda create -n environment.yml` to create a conda environment that has the needed dependencies.

Tested with Tensorflow 2.0.0, Python 3.7.9, Ubuntu 14.04. 


## Third-party pretrained networks

Our method relies on several pretrained networks.
Some are needed only for training and some also for inference.
Download according to your intention.

Put all downloaded files/directories under a single directory, which will
be the baseline path for all pretrained networks.

| Name | Training | Inference |Description
| :--- | :----------:| :----------:| :----------
|[FFHQ StyleGAN 256x256](https://drive.google.com/drive/folders/1OgLvUhd9FX9_mPXrfqAWaLZsceQzE9l4?usp=sharing) | :heavy_check_mark: | :heavy_check_mark:  | StyleGAN model pretrained on FFHQ with 256x256 resolution. Converted using [StyleGAN-Tensorflow2](https://github.com/YotamNitzan/StyleGAN-Tensorflow2)
|[FFHQ StyleGAN 1024x1024](https://drive.google.com/drive/folders/1jQxJsmapu6SjygvJfvP4-YVxZ9f5Hu_N?usp=sharing) | :heavy_check_mark: | :heavy_check_mark:  | StyleGAN model pretrained on FFHQ with 1024x1024 resolution. Converted using [StyleGAN-Tensorflow2](https://github.com/YotamNitzan/StyleGAN-Tensorflow2)
|[VGGFace2](https://drive.google.com/file/d/1I_JyR7LH-30hEIpD4OSFVg2TOf9Q8cqU/view?usp=sharing) | :heavy_check_mark: | :heavy_check_mark:  | Pretrained VGGFace2 model taken from [WeidiXie](https://github.com/WeidiXie/Keras-VGGFace2-ResNet50).
|[dlib landmarks model](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) |  | :heavy_check_mark: | dlib landmarks model, used to align images.
|[ArcFace](https://drive.google.com/drive/folders/1F-Ll9Nw7I1FGP61cpQxOdhs2nxi0E5mg?usp=sharing) | :heavy_check_mark: |   | Pretrained ArcFace model taken from [dmonterom](https://github.com/dmonterom/face_recognition_TF2).
|[Face & Landmarks Detection](https://drive.google.com/drive/folders/1D__J9UMwzBNR9eVrQGYuL9ueYGi7G4qh?usp=sharing) | :heavy_check_mark: |   | Pretrained face detection and differentiable facial landmarks detection from [610265158](https://github.com/610265158/face_landmark).


### Other StyleGANs

To try out our method with other checkpoints of StyleGAN, first obtain a trained StyleGAN pkl file using the [original StyleGAN repository](https://github.com/NVlabs/stylegan)  
Next, convert it to Tensorflow-2.0 using this [repository](https://github.com/YotamNitzan/StyleGAN-Tensorflow2).



