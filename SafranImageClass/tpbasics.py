from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import os

from ipywidgets import *

def img_numpy(img_path):
    img = load_img(img_path)
    return img_to_array(img, dtype='int')

def show_img_file(img_path, figsize=(10, 8), ax = None):
    img_array = img_numpy(img_path)
    show_img_array(img_array, img_title=img_path, ax = ax)
    
def show_img_array(img_array, figsize=(10, 8), img_title="", ax=None):
    if ax is None:
        plt.figure(figsize=figsize)
        plt.imshow(img_array)
        plt.title(img_title)
        plt.axis('off')
    else:
        ax.imshow(img_array)
        ax.set_title(img_title)
    
def elements_in_folder(folder):
    lst = []
    append_elements(folder, lst)
    return lst    
    
def n_elements_in_folder(folder):
    lst = elements_in_folder(folder)
    return len(lst)

def images_in_folder_numpy(folder):
    elements = elements_in_folder(folder)
    n_elements = len(elements)
    
    img_shape = img_numpy(elements[0][0] + "/" + elements[0][1]).shape
    
    full_array = np.zeros((n_elements,) + img_shape)
    
    for i, element in enumerate(elements):
        full_array[i, :, :, :] = img_numpy(element[0] + "/" + element[1])
        
    return full_array
    
def append_elements(folder, lst):
    elements = os.listdir(folder)
    
    for el in elements:
        if "." in el:
            lst.append((folder, el))
        else:
            append_elements(folder + "/" + el, lst)
    return lst

def animate_files_in_dir(folder, figsize):
    imgs = elements_in_folder(folder)
    n_imgs = len(imgs)
    
    def view_img(x):
        path, filename = imgs[x]
        show_img_file(path + "/" + filename, figsize = figsize)
        
    interact(view_img, x=widgets.IntSlider(min=0,max=n_imgs-1,step=1,value=0))
    
def original_patch(i, filter_sizes, strides):
    n_layers = len(filter_sizes)
    
    I_left = i
    I_right = i + 1
    
    for i_layer in range(n_layers - 1, -1, -1):
        filter_size = filter_sizes[i_layer]
        stride = strides[i_layer]
        
        I_left = stride * I_left
        I_right = (I_right - 1) * stride + filter_size
        
    return I_left, I_right

def numpy_inverse_patch(original_image, i, j, filter_sizes, strides):
    i_start, i_end = original_patch(i, filter_sizes, strides)
    j_start, j_end = original_patch(j, filter_sizes, strides)
    return original_image[i_start:i_end, j_start:j_end, :]

def numpy_max_activations(original_images, intermediate_activations, filter_sizes, strides):
    n_images = original_images.shape[0]
    n_features = intermediate_activations.shape[3]
    top_sources = []
    for i_feature in range(n_features):
        
        max_image = None
        top_act = 0
        
        for i_image in range(n_images):
            
            original_image = original_images[i_image, :, :, :]
            
            image_acts = intermediate_activations[i_image, :, :, i_feature]
            
            ind_max = np.unravel_index(np.argmax(image_acts, axis=None), image_acts.shape)
            max_act = image_acts[ind_max]
            
            if max_act > top_act:
                top_act = max_act
                max_image = numpy_inverse_patch(original_image, ind_max[0], ind_max[1], filter_sizes, strides)
                
        top_sources.append(max_image)
    return top_sources
            