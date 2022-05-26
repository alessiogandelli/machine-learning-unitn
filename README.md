# introduction to machine learning course project
The project consisted in an image retriavial task, given an image query we had too sort a gallery by the most similar images.

My first approach was only intended to understand how this kind of project should be managed, so i started watching tutorials mainly on image classification from the official keras examples and reading the tensoflow doocumentation

https://keras.io/examples/vision/image_classification_from_scratch/ 

# The siamese network
After the lessong concerning imege retrieval I discovered the existance of siamese network and i dediced that that will be the next step, in particualar i decided to implement a siamese netwoork with triplet loss even if in the very first moment i tried to implement a siamese with pairs (anchor-positive, anchor-negative) but since the scarcity oof the resources i decided t omoove on triples, in this case for each entry there are 3 instead of 4 images.


All started with with this https://keras.io/examples/vision/siamese_network/ keras tutorial. 

## Dataset 

### adopt Dataset class 
Now the problem is to fit this model with our data, and the classic flow_from_directory function was not useful because nw we had to handle triplets. So I decided to build a Dataset class for handling all the Data loading part. In this moment i was not caring about the structure of the model nor the different parameters.
This model had ram problems since all the pairs were stored in lists.

### use tensorflow dataset 
As i stated before using triplets was the best choice for efficency but also for clarity, in fact each element of the dataset is an entry while for the pairs we needed 2.
Handling triplets was impossibile with just python built in data structures, so i moved to a tf.data.Dataset object which is moore efficent and optimize the resources better. ( ram problems had been a constant during all the project). 
At this point after some empirical tests we decided that the eucledian distance was better than the cosine similarity.

### transfer learning

Using a pretrained network seem a good idea, a network trained on million of images could be useful to our task to, so we started with a classic resnet50, vgg16 and efficient net, we needed a good tradeoff between effieciency and accuracy. Our main problem at this point was testing the different models, colab free gpu is unreliable and we needed a lot of time to train all of them. So we tested on our dataset the accuracy without training and they had all similar accuracies. 

here i'm describing the structure of the embedding which is the modular part that repeated 3 times and merged with a distance layer makes a siamese network.

The first structure was a freezed resnet with 3 dense layer but since the weights of this layers are random, we risk to not have the right amount of images to correcly train it so we decided to completly remove it and just set the include_top parameter to false. But then we noticed that the includetop = False removed also the global average pooling layer so we had to add it in order to have a feature vector of a reasonable size 2048 in our case.

## Data Generator

when finally the model had a good shape, arrived the time for training, the whole dataset was to big to ofit in memory and complete an epoch, so i had to implment e custom data generator, now during the training only one batch at time was generated. Now i can train an all the 33k image dataset without problem but sadly something it's not woorking and the accuracy does not improve.



# utils 

after repeating in many files the same functiono for the evaluation and for the data loading i decided to import these methoods from a separate python file, in fact thanks to these function , given a feature extractor and the path of the gallery and the query i'm able to compute the results without adding code 