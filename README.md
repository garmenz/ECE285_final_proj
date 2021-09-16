# Explore Image Translation in Various Domains
### ECE285 final project

### Problem
The problem we try to tackle is that given an image in low information (e.g. objects are represented by color block) how do we develop a model to learn to produce natural images from this limited information. This problem is a subset of image-to-image translation, which transforms images from one domain to images from another domain with different characteristics and style. We investigate the difference in generated performance when feeding in with different type of segmented labels.

### What We Did
We organize low information image from the following three categories: semantic segmented, instance segmented and panoptic segmented image. From each of the styles, we tried to generate realistic-like image from them. Then we compare the effect of generating image with the real image. 

### Inspiration
Our model is inspired by "Image-to-Image Translation with Conditional Adversarial Networks" from Phillip Isola, Jun-Yan Zhu, TinghuiZhou, and Alexei A. Efros. 


### File Structure
- model
  - generator
  - discriminator
  - GanLoss
- data
  - img
    - rgb
    - class
    - class_3channel
    - panoptic
    - instance
  - dataset-parser
- train

### Run
train.ipynb
