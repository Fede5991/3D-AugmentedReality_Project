# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 19:40:07 2019

@author: Fede
"""
import numpy as np
import os
import binvox_rw
from tqdm import tqdm
from skimage.transform import rescale, resize, downscale_local_mean
from PIL import Image

#study of the original dataset situation
def dataset_situation():
    classes = ['airplane','bin','bag','basket','bathtub','bed','bench','bird house','library',
              'bottle','bowl','bus','dresser','camera','can','cap','car','smartphone','chair',
              'clock','keyboard','dishwashers','display','headphones','faucet','library',
              'guitar','helmet','jar','knife','lamp','laptop','loudspeaker','mail box','unknown',
              'microwave','motorbike','cup','piano','pillow','pistol','plant','scanner','telecontrol',
              'rifle','rocket','skate','sofa','oven','table','smartphone','tower','railway engine',
              'ship','washing machine']
    np.save('classes',classes)
    lengths=[]
    screens=[]
    Screenshots = {}
    categories = os.listdir('G:\ShapeNetCore.v2')#total number of classes
    for i in tqdm(range(len(categories))):#for all the classe I use all models
        model = os.listdir('G:\ShapeNetCore.v2/'+categories[i])
        l=0
        s=0
        for j in range(len(model)):
                try:
                    with open('G:\ShapeNetCore.v2/'+categories[i]+'/'+model[j]+'/models/'+'model_normalized.surface.binvox', 'rb') as f:
                        surface = binvox_rw.read_as_coord_array(f)
                        l+=1
                        try:
                            screenshots=os.listdir('G:\ShapeNetCore.v2/'+categories[i]+'/'+model[j]+'/screenshots')
                            s+=len(screenshots)
                            if len(screenshots) in Screenshots:
                                Screenshots[len(screenshots)]+=1
                            else:
                                Screenshots[len(screenshots)]=1
                        except:
                            pass
                except:
                    pass
        lengths.append(l)
        screens.append(s)
    return classes,lengths,screens,min(lengths)

def creation_datasets(maximum,ratio,ratio_images,reduction,dataset_classification,
                      dataset_autoencoder,noise,from3dto2d):
    training_autoencoder = []
    labels_autoencoder = []
    models_number=maximum
    dataset = []
    twoimages=[]
    categories = os.listdir('G:\ShapeNetCore.v2')#total number of classes
    for i in tqdm(range(len(categories))):
        model = os.listdir('G:\ShapeNetCore.v2/'+categories[i])
        l=0
        for j in range(56):
    
                try:
                    with open('G:\ShapeNetCore.v2/'+categories[i]+'/'+model[j]+
                              '/models/'+'model_normalized.surface.binvox', 'rb') as f:
                        surface = binvox_rw.read_as_coord_array(f)
                        original_side = 128
                        new_side = int(original_side/ratio)
                        Volume = np.zeros((original_side,original_side,original_side))
                        for k in range(len(surface.data[0])):
                            x = surface.data[0][k]
                            y = surface.data[1][k]
                            z = surface.data[2][k]
                            Volume[x,y,z]=1
                        if from3dto2d==True:
                            scan=np.zeros((Volume.shape[0],Volume.shape[0]))
                            for x1 in range(Volume.shape[0]):
                                for y1 in range(Volume.shape[0]):
                                    for z1 in range(Volume.shape[0]):
                                        if Volume[x1,y1,z1]>0:
                                            scan[x1,y1]=1
                                            z1=128
                            twoimages.append(scan)
                            scan=np.zeros((Volume.shape[0],Volume.shape[0]))
                            for y1 in range(Volume.shape[0]):
                                for z1 in range(Volume.shape[0]):
                                    for x1 in range(Volume.shape[0]):
                                        if Volume[x1,y1,z1]>0:
                                            scan[y1,z1]=1
                                            x1=128
                            twoimages.append(scan)
                            scan=np.zeros((Volume.shape[0],Volume.shape[0]))
                            for z1 in range(Volume.shape[0]):
                                for x1 in range(Volume.shape[0]):
                                    for y1 in range(Volume.shape[0]):
                                        if Volume[x1,y1,z1]>0:
                                            scan[z1,x1]=1
                                            y1=128
                            twoimages.append(scan)
                        if reduction==True:
                            Volume_resized = resize(Volume, (new_side,new_side,new_side),anti_aliasing=True)
                            if noise==True:
                                for d1 in range(Volume_resized.shape[0]):
                                    for d2 in range(Volume_resized.shape[0]):
                                        for d3 in range(Volume_resized.shape[0]):
                                            if np.random.rand(1)[0]<0.001:
                                                Volume_resized[d1,d2,d3]+=0.1
                            if l<models_number and dataset_classification==True:
                                dataset.append(Volume_resized)
                        else:
                            if l<models_number and dataset_classification==True:
                                dataset.append(Volume)
                        l+=1
                        if dataset_autoencoder==True:
                            try:
                                screenshots=os.listdir('G:\ShapeNetCore.v2/'+categories[i]+'/'+model[j]+
                                  '/screenshots')
                                if len(screenshots)==14:
                                    count=0
                                    new_size_image = int(512/ratio_images)
                                    images=np.zeros((new_size_image,new_size_image,14))
                                    for screen in screenshots:
                                        img = Image.open('G:/ShapeNetCore.v2/'+categories[i]+'/'+model[j]+
                                                          '/screenshots/'+screen)
                                        array = np.asarray(img)
                                        image = array[:,:,0]/255
                                        image = resize(image,(new_size_image,new_size_image),anti_aliasing=True)
                                        images[:,:,count]=image
                                        count+=1
                                    training_autoencoder.append(images)
                                    labels_autoencoder.append(Volume_resized)
                            except:
                                pass
                        
                except:
                    pass
    
    if dataset_classification==True:
        training_samples = 55*56
        labels = np.zeros((training_samples,55))
        for i in range(training_samples):
            j=int(i/56)
            labels[i,j]=1
        np.save('labels',labels)
        np.save('dataset',dataset)
        if from3dto2d==True:
            np.save('2dimages',twoimages)
    
    if dataset_autoencoder==True:    
        np.save('training_autoencoder',training_autoencoder)
        np.save('labels_autoencoder',labels_autoencoder)