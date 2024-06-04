# Introduction to Machine Learning Course Project

This project was an image retrieval task. Given an image query, we sorted a gallery by the most similar images.

My initial approach aimed to understand how to manage this type of project. I began by watching tutorials on image classification from the official Keras examples and reading the TensorFlow documentation.

https://keras.io/examples/vision/image_classification_from_scratch/

# The Siamese Network

After the lesson on image retrieval, I learned about the Siamese network. I decided that this would be the next step. In particular, I chose to implement a Siamese network with triplet loss. Initially, I tried to implement a Siamese network with pairs (anchor-positive, anchor-negative) but due to resource limitations, I decided to use triples. In this case, for each entry, there are three images instead of four.

My journey began with this Keras tutorial: https://keras.io/examples/vision/siamese_network/

## Dataset

### Adopt Dataset Class

Fitting this model with our data presented a challenge, as the classic flow_from_directory function was not suitable for handling triplets. I built a Dataset class to manage all the data loading. This model had RAM limitations as all pairs were stored in lists.

### Use TensorFlow Dataset

Using triplets was the most efficient and clear choice. Each element of the dataset is an entry, whereas for pairs, we needed two. Handling triplets with just Python's built-in data structures was impossible, so I moved to a tf.data.Dataset object which is more efficient and better optimizes resources. After running some tests, we found that Euclidean distance was better than cosine similarity.

### Transfer Learning

We tried using a pre-trained network, such as ResNet50, VGG16, and EfficientNet. We needed a balance between efficiency and accuracy. Our main problem was testing the different models. Colab's free GPU is unreliable and we needed a lot of time to train all of them.

The structure of the embedding, which is the modular part repeated three times and merged with a distance layer, forms a Siamese network. The initial structure was a frozen ResNet with three dense layers. To have a feature vector of a reasonable size (2048 in our case), we added a global average pooling layer.

## Data Generator

When the model was in good shape, we moved to training. The entire dataset was too large to fit in memory and complete an epoch, so I implemented a custom data generator. Now, during training, only one batch at a time was generated. However, the accuracy did not improve.

# Utils

After repeating many functions for evaluation and data loading in multiple files, I decided to import these methods from a separate file. Now, given a feature extractor and the path of the gallery and query, I can compute the results without adding code.
