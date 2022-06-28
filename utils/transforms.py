import torch
import copy
from monai.transforms import Randomizable, MapTransform
from monai.config import KeysCollection
import numpy as np

import math


class RandPatchedImageWith0Padding(Randomizable, MapTransform):
    
    def __init__(self, keys: KeysCollection, num_patches=9):
        super().__init__(keys)
        self.num_patches = num_patches
        self.start_id = 0
        self.num_cols = int(math.sqrt(self.num_patches))
        self.dim = 1
        assert (self.num_cols**2 == self.num_patches), 'Error in num_slices values'
    
    def randomize(self, max_value):
        self.start_id = int(self.R.random_sample() * max_value)
        
    def __call__(self, data):
        img_key, label_key = self.keys
        patch_row = data[img_key].size()[2]
        patch_col = data[img_key].size()[3]
        patched_img = torch.zeros(data[img_key].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        num_slices = data[img_key].size()[1]
        
        if num_slices < self.num_patches:
            self.start_id = 0
        else: self.randomize(num_slices - self.num_patches -1 )
        
        counter = 0
        for i in range(self.num_cols):
            for j in range(self.num_cols):
                if counter < num_slices:
                    patched_img[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key][:,self.start_id+counter,:,:]
                    counter += 1
       
        cropper = copy.deepcopy(data)
        cropper[img_key] = patched_img
        cropper['start_id'] = self.start_id
        return cropper
    
class RandPatchedImage(Randomizable, MapTransform):
    
    def __init__(self, keys: KeysCollection, num_patches=9):
        super().__init__(keys)
        self.num_patches = num_patches
        self.start_id = 0
        self.num_cols = int(math.sqrt(self.num_patches))
        self.dim = 1
        assert (self.num_cols**2 == self.num_patches), 'Error in num_slices values'
    
    def randomize(self, max_value):
        self.start_id = int(self.R.random_sample() * max_value)
        
    def __call__(self, data):
        img_key, label_key = self.keys
        patch_row = data[img_key].size()[2]
        patch_col = data[img_key].size()[3]
        patched_img = torch.zeros(data[img_key].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        num_slices = data[img_key].size()[1]
        
        if num_slices < self.num_patches:
            self.start_id = 0
        else: self.randomize(num_slices - self.num_patches -1 )
        
        counter = 0
        for i in range(self.num_cols):
            for j in range(self.num_cols):
                if counter < num_slices:
                    patched_img[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key][:,self.start_id+counter,:,:]
                    counter += 1
                else:
                    patched_img[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key][:,self.start_id+counter-1,:,:]
       
        cropper = copy.deepcopy(data)
        cropper[img_key] = patched_img
        cropper['start_id'] = self.start_id
        return cropper

class RandPatchedImageLateFusion(Randomizable, MapTransform):
    def __init__(self, keys: KeysCollection, num_patches=9):
        super().__init__(keys)
        self.num_patches = num_patches
        self.start_id = 0
        self.num_cols = int(math.sqrt(self.num_patches))
        self.dim = 1
        assert (self.num_cols**2 == self.num_patches), 'Error in num_slices values'
    
    def randomize(self, max_value):
        self.start_id = int(self.R.random_sample() * max_value)
        
    def __call__(self, data):
        img_key_T1, img_key_T2, label_key = self.keys
        patch_row = data[img_key_T1].size()[2]
        patch_col = data[img_key_T1].size()[3]
        patched_img_T1 = torch.zeros(data[img_key_T1].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        patched_img_T2 = torch.zeros(data[img_key_T1].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        num_slices = data[img_key_T1].size()[1]
        
        if num_slices < self.num_patches:
            self.start_id = 0
        else: self.randomize(num_slices - self.num_patches -1 )
        
        counter = 0
        for i in range(self.num_cols):
            for j in range(self.num_cols):
                if counter < num_slices:
                    patched_img_T1[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T1][:,self.start_id+counter,:,:]
                    patched_img_T2[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T2][:,self.start_id+counter,:,:]
                    counter += 1
                else:
                    patched_img_T1[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T1][:,self.start_id+counter-1,:,:]
                    patched_img_T2[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T2][:,self.start_id+counter-1,:,:]
       
        cropper = copy.deepcopy(data)
        cropper[img_key_T1] = patched_img_T1
        cropper[img_key_T2] = patched_img_T2
        cropper['start_id'] = self.start_id
        return cropper

class RandPatchedImageAndEarlyFusion(Randomizable, MapTransform):
    
    def __init__(self, keys: KeysCollection, num_patches=9):
        super().__init__(keys)
        self.num_patches = num_patches
        self.start_id = 0
        self.num_cols = int(math.sqrt(self.num_patches))
        self.dim = 1
        assert (self.num_cols**2 == self.num_patches), 'Error in num_slices values'
    
    def randomize(self, max_value):
        self.start_id = int(self.R.random_sample() * max_value)
        
    def __call__(self, data):
        img_key_T1, img_key_T2, label_key = self.keys
        patch_row = data[img_key_T1].size()[1]
        patch_col = data[img_key_T1].size()[2]
        patched_img_T1 = torch.zeros(data[img_key_T1].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        patched_img_T2 = torch.zeros(data[img_key_T1].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        num_slices = data[img_key_T1].size()[3]
        
        if num_slices < self.num_patches:
            self.start_id = 0
        else: self.randomize(num_slices - self.num_patches -1 )
        
        counter = 0
        for i in range(self.num_cols):
            for j in range(self.num_cols):
                if counter < num_slices:
                    patched_img_T1[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T1][:,:,:,self.start_id+counter]
                    patched_img_T2[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T2][:,:,:,self.start_id+counter]
                    counter += 1
                else:
                    patched_img_T1[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T1][:,:,:,self.start_id+counter-1]
                    patched_img_T2[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T2][:,:,:,self.start_id+counter-1]
       
        patched_fused_img = torch.cat((patched_img_T1, patched_img_T2), dim = 0)
        cropper = copy.deepcopy(data)
        del cropper[img_key_T1]
        del cropper[img_key_T2]
        cropper['fusedImage'] = patched_fused_img
        cropper['start_id'] = self.start_id
        return cropper

class CenterPatchedImageAndEarlyFusion(MapTransform):
    
    def __init__(self, keys: KeysCollection, num_patches=9):
        super().__init__(keys)
        self.num_patches = num_patches
        self.start_id = 0
        self.num_cols = int(math.sqrt(self.num_patches))
        self.dim = 1
        assert (self.num_cols**2 == self.num_patches), 'Error in num_slices values'
        counter = 0
    
      
    def __call__(self, data):
        img_key_T1, img_key_T2, label_key = self.keys
        patch_row = data[img_key_T1].size()[1]
        patch_col = data[img_key_T1].size()[2]
        patched_img_T1 = torch.zeros(data[img_key_T1].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        patched_img_T2 = torch.zeros(data[img_key_T1].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        num_slices = data[img_key_T1].size()[3]
        
        if num_slices < self.num_patches:
            self.start_id = 0
        else:
            self.start_id = int(num_slices/2)-int(self.num_patches/2)
        
        counter = 0
        for i in range(self.num_cols):
            for j in range(self.num_cols):
                if counter < num_slices:
                    patched_img_T1[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T1][:,:,:,self.start_id+counter]
                    patched_img_T2[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T2][:,:,:,self.start_id+counter]
                    counter += 1
                else:
                    patched_img_T1[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T1][:,:,:,self.start_id+counter-1]
                    patched_img_T2[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T2][:,:,:,self.start_id+counter-1]
       
        patched_fused_img = torch.cat((patched_img_T1, patched_img_T2), dim = 0)
        cropper = copy.deepcopy(data)
        del cropper[img_key_T1]
        del cropper[img_key_T2]
        cropper['fusedImage'] = patched_fused_img
        cropper['start_id'] = self.start_id
        return cropper
        

class RandDepthCrop(Randomizable, MapTransform):
    
    def __init__(self, keys: KeysCollection, num_slices=3):
        super().__init__(keys)
        self.num_slices = num_slices
        self.start_id = 0
    
    def randomize(self, max_value):
        self.start_id = int(self.R.random_sample() * max_value)
        
    def __call__(self, data):
        img_key, label_key = self.keys
        max_value = data[img_key].shape[1] - self.num_slices
        self.randomize(max_value)
        slice_ = data[img_key][0,self.start_id:(self.start_id+self.num_slices),:,:]
        n = slice_.shape[0]
        while n<self.num_slices:
            slice_ = torch.cat([slice_, slice_[-1].unsqueeze(0)],dim = 0)
            n+=1
        cropper = copy.deepcopy(data)
        cropper[img_key] = slice_
        cropper['start_id'] = self.start_id
        return cropper
    
class NewMergedImage(MapTransform):
    def __init__(self, keys: KeysCollection, num_patches=9):
        super().__init__(keys)
    
    def __call__(self, data):
        img_key_T1, img_key_T2 = self.keys
        img_key_merge = 'merged'
        
        merged_data = data[img_key_T1] - data[img_key_T2]
        data[img_key_merge] = merged_data
        return data
    
class RandPatchedImage3Channels(Randomizable, MapTransform):
    
    def __init__(self, keys: KeysCollection, num_patches=9):
        super().__init__(keys)
        self.num_patches = num_patches
        self.start_id = 0
        self.num_cols = int(math.sqrt(self.num_patches))
        self.dim = 1
        assert (self.num_cols**2 == self.num_patches), 'Error in num_slices values'
    
    def randomize(self, max_value):
        self.start_id = int(self.R.random_sample() * max_value)
        
    def __call__(self, data):
        img_key_T1, img_key_T2, label_key = self.keys
        img_key_merge = 'merged'
        patch_row = data[img_key_T1].size()[2]
        patch_col = data[img_key_T1].size()[3]
        patched_img_T1 = torch.zeros(data[img_key_T1].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        patched_img_T2 = torch.zeros(data[img_key_T1].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        patched_merge = torch.zeros(data[img_key_T1].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        num_slices = data[img_key_T1].size()[1]
        
        if num_slices < self.num_patches:
            self.start_id = 0
        else: self.randomize(num_slices - self.num_patches -1 )
        
        counter = 0
        for i in range(self.num_cols):
            for j in range(self.num_cols):
                if counter < num_slices:
                    patched_img_T1[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T1][:,self.start_id+counter,:,:]
                    patched_img_T2[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T2][:,self.start_id+counter,:,:]
                    patched_merge[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_merge][:,self.start_id+counter,:,:]
                    counter += 1
                else:
                    patched_img_T1[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T1][:,self.start_id+counter-1,:,:]
                    patched_img_T2[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T2][:,self.start_id+counter-1,:,:]
                    patched_merge[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_merge][:,self.start_id+counter-1,:,:]
                    
        patched_fused_img = torch.cat((patched_img_T1, patched_img_T2, patched_merge), dim = 0)
        cropper = copy.deepcopy(data)
        del cropper[img_key_T1]
        del cropper[img_key_T2]
        del cropper[img_key_merge]
        cropper['fusedImage'] = patched_fused_img
        cropper['start_id'] = self.start_id
        return cropper
    
class NDITKtoNumpy(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)
        
    def __call__(self, data):
        for k in self.keys:
            data[k] = np.asarray(data[k])
        return data