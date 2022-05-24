# -*- coding: utf-8 -*-
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import random
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from PIL import Image
import imghdr
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications import resnet
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.python.keras.utils.data_utils import Sequence



class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_path , batch_size=32, num_classes=None, shuffle=True, target_shape = (128,128)):
        print('init generator')
        #path of the dataset
        self.data_path = data_path
        self.batch_size= batch_size
        self.target_shape = target_shape
        
        #class list
        self.data_classes = [directory for directory in os.listdir(data_path) if os.path.isdir(data_path+directory)]

        # init lists and dictionary
        self.images = []
        self.labels = []


        # for each class and for each image save the image and the label in the lists 
        for c, c_name in enumerate(self.data_classes):
            temp_path = os.path.join(self.data_path, c_name)
            temp_images = os.listdir(temp_path)

            for i in temp_images:
                img_tmp = os.path.join(temp_path, i)

                if img_tmp.endswith('.jpg') or img_tmp.endswith('.JPEG') or img_tmp.endswith('.JPG') or img_tmp.endswith('.jpeg') :  
                  self.images.append(img_tmp)
                  self.labels.append(c_name)
                
        self.indexes = list(range(len(self.images)))    
        random.shuffle(self.indexes)

        print('Loaded {:d} images from {:s} '.format(len(self.images), self.data_path))

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def __getitem__(self, index):
        batch = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        
        (a,p,n)= self.__make_triplets(batch)
        
        
        return (a,p,n)

    def on_epoch_end(self):
      pass


    def get_random_image_idx_same_class(self, classe):

        label = 'formaggio'

        while label != classe:
            idx = np.random.choice(len(self.images))
            label = self.labels[idx]
        
        return idx

    def get_random_image_idx_different_class(self, classe):

        label = classe

        while label == classe:
            idx = np.random.choice(len(self.images))
            label = self.labels[idx]
        return idx
    
    def __make_triplets(self, batch):


          anchors = []
          positive = []
          negative = []

          # an array of size (batch_size, image_size)
          a = np.empty((self.batch_size, *(self.target_shape+ (3,)) ))
          p = np.empty((self.batch_size, *(self.target_shape+ (3,)) ))
          n = np.empty((self.batch_size, *(self.target_shape+ (3,)) ))


          
          
          for i, img in enumerate(batch):
              #current image 
              anc_img = self.images[img]
              anchors.append(anc_img)
              currentLabel = self.labels[img]

              #positive image 
              pos_idx = self.get_random_image_idx_same_class(currentLabel)
              pos_img = self.images[pos_idx]
              positive.append(pos_img)

              #negative image
              neg_idx = self.get_random_image_idx_different_class(currentLabel)
              neg_img = self.images[neg_idx]
              negative.append(neg_img)
              
              a[i,] = np.array((self.preprocess_image(anc_img)))
              p[i,] = np.array((self.preprocess_image(pos_img)))
              n[i,] = np.array((self.preprocess_image(neg_img)))


          return (a,p,n)
    
    # from filename to image tensor 
    def preprocess_image(self, filename):

        image_string = tf.io.read_file(filename)
        #image = tf.image.decode_jpeg(image_string, channels=1)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, self.target_shape)
        return image


    def get_dataset(self):
        names = tf.data.Dataset.from_tensor_slices(self.images)
        ds = names.map(self.preprocess_image)
        labels = tf.data.Dataset.from_tensor_slices(self.labels)
        return tf.data.Dataset.zip((names,ds,labels)).shuffle(buffer_size=1024).batch(self.batch_size).prefetch(8)

class Dataset(object):
    def __init__(self, data_path, target_shape = (128,128), batch_size = 64):
      #path of the dataset
      self.data_path = data_path
      self.batch_size = batch_size
      self.target_shape = target_shape


      #class list
      self.data_classes = [directory for directory in os.listdir(data_path) if os.path.isdir(data_path+directory)]

      # init lists and dictionary
      self.images = []
      self.labels = []


      # for each class and for each image save the image and the label in the lists 
      for c, c_name in enumerate(self.data_classes):
          temp_path = os.path.join(self.data_path, c_name)
          temp_images = os.listdir(temp_path)

          for i in temp_images:
              img_tmp = os.path.join(temp_path, i)

              if img_tmp.endswith('.jpg') or img_tmp.endswith('.JPEG') or img_tmp.endswith('.JPG') or img_tmp.endswith('.jpeg') :  
                self.images.append(img_tmp)
                self.labels.append(c_name)
                
                  
      print('Loaded {:d} images from {:s} '.format(len(self.images), self.data_path))

    def num_classes(self):
        # returns number of classes of the dataset
        return len(self.data_classes)
    
    def get_dataset(self):
        names = tf.data.Dataset.from_tensor_slices(self.images)
        ds = names.map(self.preprocess_image)
        labels = tf.data.Dataset.from_tensor_slices(self.labels)
        return tf.data.Dataset.zip((names,ds,labels)).shuffle(buffer_size=1024).batch(self.batch_size).prefetch(8)
    
    def get_images(self):
        
        return list(map(self.preprocess_image, self.images))


    def preprocess_image(self, filename):
        """
        Load the specified file as a JPEG image, preprocess it and
        resize it to the target shape.
        """
        image_string = tf.io.read_file(filename)
        #image = tf.image.decode_jpeg(image_string, channels=1)
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, self.target_shape)
        return image



def compute_features(ds, embedding):
    features = []
    paths = []
    labs = {} # this is for the evaluation part, labs[/data/dog.jpg] = dog
    # for all batches of the dataset (query or gallery)
    for names, imgs, labels in ds:
        embeddings = embedding(imgs) # calculate feature vectors of a batch of images 
        # for all embeddings in a batch
        for emb in range(len(embeddings)):
            path = bytes.decode(names[emb].numpy()) # extract path 
            #append to lists 
            features.append(embeddings[emb]) 
            paths.append(path)
            labs[path] = bytes.decode(labels[emb].numpy())

    return features, paths, labs

def compute_results(query_features, gallery_features, query_urls, gallery_urls ):
  results = dict()
  print(len(query_features))
  for i, query_sample in enumerate(query_features):

      query_sample = tf.reshape(query_sample, [1, len(query_sample)])
      query_sample_tiles = tf.tile(query_sample, [len(gallery_features), 1])
      dists = tf.math.sqrt(tf.reduce_sum(tf.math.square(query_sample_tiles - gallery_features), axis=-1))
      rank = tf.argsort( dists, axis=-1, direction='ASCENDING', stable=False, name=None)
      gallery_list = []
      for idx in rank[:10]:
        gallery_list.append(gallery_urls[idx])
      
      results[query_urls[i]] = gallery_list
   
          
  return results

def evaluate(results, query_labels, gallery_labels):
    top = {1:0, 3:0, 5:0, 10:0 , 11:0}
    top_norm = {}

    for query, gallery in results.items():
      q_lab = query_labels[query]
      
      res = next((i for i, x in enumerate(gallery) if q_lab == gallery_labels[x]), 10) +1

      for i in [1,3,5,10]:
        if res <= i:
          top[i] += 1

      if res == 11:
        top[11] += 1

      tot = top[10] + top[11]

      for key, value in top.items():
        top_norm[key] = round(value/tot, 2)
    
    return top_norm


def clean_dataset(path):
    data_classes = [directory for directory in os.listdir(path) if os.path.isdir(path+directory)]

    for c, c_name in enumerate(data_classes):
        temp_path = os.path.join(path, c_name)
        temp_images = os.listdir(temp_path)

        for i in temp_images:
            img_tmp = os.path.join(temp_path, i)
            ext = imghdr.what(img_tmp)
           
            if ext not in ['jpeg', 'png']:
              pass
              #os.remove(img_tmp)
              print(ext, img_tmp)