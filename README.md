# Race Detection with RetinaNet
This kernel involves using a Pretrained RetinaNet Model and fine-tuning it to classify and detect (to draw a bounding box around the object of a specific race) different Races such as Mongoloid, Negroid and Caucasian.

But first of all, I will relate the basic intuition behind RetinaNet

![5d30d50bbe811865302eda60_s_9F0961CE65C906E6A73F32E5A1E42780840414967C63EDB3B15BF25F424BC7C0_1563480199678_image](https://user-images.githubusercontent.com/51057490/82675735-296a6c80-9c3d-11ea-8f02-d84e808f8e3f.png)

**RetinaNet** is the current **state-of-the-art Object Detector Algorithm** which is composed of three networks
- A **backbone network** called Feature Pyramid Net (FPN), which is built on top of ResNet
- A **subnetwork** responsible for performing object classification using the backbone’s output
- A **subnetwork** responsible for performing bounding box regression using the backbone’s output

It is also a One-stage detector such as YOLO, SSD, Overfeat etc, but it varies by producing a feature map that is **both semantically and spatially strong**, (i.e, low resolution images which are spatially strong is combined
with it's correspondent which is semantically strong and high resolution images which are semantically strong but spatially coarse are combined with it's correspondent which is spatially strong) using lateral connection.

Also RetinaNet solves the major problem of One-stage detectors, which is **compromising accuracy for speed** - this is the main reason why major Two-stage detectors such as Faster R-CNN are preferred when the feat of a project is aimed at accuracy rather than speed.

What of if the feat of the project is aimed at both accuracy and speed? *RetinaNet to the answer...*

[Tsung-Yi Lin proprosed a Dense Object Detection](https://arxiv.org/abs/1708.02002) (i.e having more boxes to densely cover the space of all possible objects) in order to improve the accuracy of one-stage detectors.

Compared to existing One-stage detectors, RetinaNet provided more boxes
- YOLOv1 - 98 boxes
- YOLOv2 - ~1k boxes
- Overfeat - ~1-2k boxes
- SSD - ~8-26k boxes
- RetinaNet - ~100k boxes

*But Hold on...*

This caused a major problem as a result of the class imbalance - as there were more training examples from background which were easy, disinformative and distracting and just few examples from foreground (where the actual object is) which contained rich information

Class imbalance such as having 7 objects in an image whereas we have 100k boxes location.

Due to the large number of training examples from background, this impacted on the loss function and also failed the training process. This is because hard examples (foreground) contain more informative signal which will be overwhelmed by the lost generated from the easy examples

Hence RetinaNet adopts a Cross Entropy modulated function which reduces the effect of the loss for easy examples whereas keeps the loss for hard examples, called [Focal loss](https://arxiv.org/abs/1708.02002).


As earlier stated, this kernel involves using a Pretrained RetinaNet to train a Model to classify and detect (to draw a bounding box around the object of a specific race) Different Races such as Mongoloid, Negroid and Caucasian

Each of the object in the image are either labelled as:

- Caucasian: includes people of American and European descent, also known as whites

- Mongoloid: includes people of Asian descent, especially Eastern Asian

- Negroid: includes people of African descent or black Americans

The *workflow* are outlined as follows:
- Use [labelimg](https://github.com/tzutalin/labelImg) to annotate (label and specify the bounding box cordinates) all the objects in the image in a Pascal VOC format
- Run the xml_script.py to convert the Pascal VOC format (xml) to csv as that's what a RetinaNet expects
- Load the Pretrained RetinaNet from keras and all it's dependencies and navigate to the main file directory
- Train the Pretrained RetinaNet by specifying a backbone (I used Retina50) and save the learned parameters after each epochs
- Convert the saved model to an inference graph to test on unseeen data

Here are some of following Results from the Inference Model:


![view7](https://user-images.githubusercontent.com/51057490/82677218-954dd480-9c3f-11ea-8dcd-3236faaa412e.JPG)
![view9](https://user-images.githubusercontent.com/51057490/82677230-9a128880-9c3f-11ea-9c78-e6735851c69e.JPG)
![view8](https://user-images.githubusercontent.com/51057490/82677237-9d0d7900-9c3f-11ea-929d-ff8181c12421.JPG)
![view11](https://user-images.githubusercontent.com/51057490/82677247-a1399680-9c3f-11ea-8b25-a09af8fe509a.JPG)
![view2](https://user-images.githubusercontent.com/51057490/82677259-a4348700-9c3f-11ea-946f-2a9488733185.JPG)
![view3](https://user-images.githubusercontent.com/51057490/82677364-ce864480-9c3f-11ea-9dd4-2be84a86ba35.JPG)
![view1](https://user-images.githubusercontent.com/51057490/82677381-d47c2580-9c3f-11ea-83b4-46769613ad9b.JPG)
![view6](https://user-images.githubusercontent.com/51057490/82677396-dc3bca00-9c3f-11ea-93e7-474543076f60.JPG)
![view10](https://user-images.githubusercontent.com/51057490/82677415-e4940500-9c3f-11ea-864e-4d6b74d09955.JPG)
![view5](https://user-images.githubusercontent.com/51057490/82677608-32a90880-9c40-11ea-830a-f37b72aac3e8.JPG)
![view4](https://user-images.githubusercontent.com/51057490/82677600-2fae1800-9c40-11ea-86ce-d449a64376fc.JPG)



# MOTIVATION
Conventionally, Neural Networks (NN) are always feed images of different shapes, textures and disimilar features and it does well in learning each
mapping or vector space relative to the others. That is NN are very good at learning mappings of nuanced distribution

But what about feeding a Neural Network images of the same shape and a kind of similar features...
How will the Neural Networks learn this less nuanced distribution?
