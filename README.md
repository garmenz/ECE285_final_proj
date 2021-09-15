# Explore Image Translation in Various Domains
### ECE285 final project

### Problem
The problem we try to tackle is that given an image in low information (e.g. objects are represented by color block) how do we develop a model to learn to produce natural images from this limited information. This problem is a subset of image-to-image translation, which transforms images from one domain to images from another domain with different characteristics and style. We investigate the difference in generated performance when feeding in with different type of segmented labels.

### What We Did
We organize low information image from the following three categories: semantic segmented, instance segmented and panoptic segmented image. 

### Inspiration
Our model is inspired by ...


### File Structure
- model
  - generator
  - discriminator
  - loss
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
