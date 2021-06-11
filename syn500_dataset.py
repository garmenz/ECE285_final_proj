from pathlib import Path
import numpy as np
import random
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
def load_samples(root,main = 'instance',search = ['rgb','class_3channel', 'panoptic']):
    root = Path(root)
    mainP =  root/main
    src = {'instance':[],'rgb':[],'class_3channel':[], 'panoptic':[]}
    all_path = [x for x in mainP.iterdir()]
    num_files = len(all_path)
    #get all files path
    print(f'Loadding files from {root}')
    for i,x in enumerate(all_path):
        if x.is_file():
            src[main].append(x)
            
            for split in search:
                src[split].append(x.parents[1]/split/x.name)
    print(f'Loaded:{num_files}')
    return src

class SegmentLoader(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:
    Addtional segmentation label is expected as well

    .. code-block::

        - rootdir/
            - img/
                -rbg/
                    - img000.png
                    - img001.png
                -instance/
                    - img000.png
                    - img001.png
                -class/
                    - img000.png
                    - img001.png

    Args:
        root (tuple): (root directory of the rgb image dataset,root directory of the segmentation label)
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
    """

    def __init__(self, root, main = 'rgb',search = ['instance','class_3channel', 'panoptic'],def_int = 'instance'):
        self.samples =load_samples(root,main =main,search =search)
        self.def_int = def_int
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        rgb_img = Image.open(self.samples['rgb'][index]).convert("RGB")
        class_img =Image.open(self.samples[self.def_int][index]).convert("RGB")
        B,A = self.transform(rgb_img,class_img)
#         if inst:
#             instance_img = np.array(Image.open(self.samples['instance'][index]).convert("RGB"))
#             stacked = torch.cat([transforms.ToTensor()(rgb_img), transforms.ToTensor()(class_img),transforms.ToTensor()(instance_img)], dim=0)
#             stacked = self.transform(stacked)
#             rgb_img, class_img,instance_img = torch.chunk(stacked, chunks=3, dim=0)
#             return rgb_img, class_img,instance_img
#         else:
#         stacked = torch.cat([transforms.ToTensor()(rgb_img), transforms.ToTensor()(class_img)], dim=0)
#         stacked = self.transform(stacked)
#         rgb_tensor, class_tensor = torch.chunk(stacked, chunks=2, dim=0)
        
#         print(f'shape{rgb_img.shape,class_img.shape,A.shape,B.shape}')
#         print(f'shape{rgb_img.shape,class_img.shape,rgb_tensor.shape,class_tensor.shape}')
        return {'A': A, 'B': B, 'A_paths': str(self.samples['class_3channel'][index]), 'B_paths': str(self.samples['rgb'][index])}

    def __len__(self):
        return len(self.samples['rgb'])
    def transform(self, image, mask):
            # Resize
            resize = transforms.Resize(size=(286, 286))
            normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            image = resize(image)
            mask = resize(mask)

            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(256, 256))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Transform to tensor
            image = TF.to_tensor(image)
            mask = TF.to_tensor(mask)
            image = normalize(image)
            mask = normalize(mask)   
            return image, mask
