# Disease Indentification on Cassava Leaf:
#### In this our task is to classify each cassava plant image into four disease categories or a fifth category indicating a healthy leaf. With this, farmers may be able to quickly identify diseased plants, potentially saving their crops before they inflict irreparable damage.

#### Categories
- Cassava Bacterial Blight (CBB)
- Cassava Brown Streak Disease (CBSD)
- Cassava Green Mottle (CGM)
- Cassava Mosaic Disease (CMD)
- "Healthy

![image](https://github.com/amancrackpot/ImageClassification_Leaf_Disease/blob/master/Results/target_dist.png)

#### The code needed to train the model is detailed in here https://nbviewer.jupyter.org/github/tripathiGithub/ImageClassification_Leaf_Disease/blob/master/cassava-tensorflow-resnet50.ipynb
(Use this link only to see the code instead of using ipynb from github directly becacuse github does not renders ipynb files properly)


Deep Learning has been used to create this model using Tensorflow. I have used Transfer Learning which involves loading a generic well trained image classification model for feature extraction and then adding a few layers as head so that it can be trained for our specific task. Apart from this, to train the system and get better results, modern deep learning practices have been used like data-augmentation , one-cycle-policy, discriminative-learning-rate, etc

#### Dataset available at https://www.kaggle.com/c/cassava-leaf-disease-classification
## Results on test data
![image](https://github.com/amancrackpot/ImageClassification_Leaf_Disease/blob/master/Results/cr.png)
![image](https://github.com/amancrackpot/ImageClassification_Leaf_Disease/blob/master/Results/cm.png)
