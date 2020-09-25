# DLP-Lab6
Let’s Play GANs
(The report please refer to DLP_LAB6_Report_0886035.pdf) 

#### Lab Objective
* In this lab, you need to implement a conditional GAN and generate synthetic images based on multi-labels conditions
* Example of labels: [“cyan cylinder”, “red cube”], [“green sphere”], …

![Objective](/picture/Objective.png "Objective")



#### Lab Description
* Implementation details
  * Choose your conditional GAN architecture
  * Design your generator and discriminator
  * Choose your loss function



#### Choice of cGAN
* Generator
  * Concatenation, multiplication, batch normalization, etc.
  * Similar technique used in lab5
* Discriminator
  * Conditional GAN
  * InfoGAN
  * Auxiliary GAN
  * Projection discriminator
* Hybrid version

![Choice of cGAN](/picture/Choice_of_cGAN.png "Choice of cGAN")


#### Design of GAN
* De-convolution layers
* Basic block
* Bottleneck block
* Residual
* Self-attention
* Again, hybrid version
* E.g.
  * DCGAN
  * SA-GAN
  * Progressive GANA



#### Choice of loss functions
* GAN loss function
  * ![Choice of loss functions](/picture/Choice_of_loss_functions.png "Choice of loss functions")
* LSGAN
* WGAN
* WGAN-GP

* Combine with your choice of cGAN


#### Other details
* You can use any GAN architecture your like
* Use the function of a pretrained classifier, eval(images, labels), to compute accuracy of your synthetic images.
  * Labels should be one-hot vector. E.g. [[1,1,0,0,…],[0,1,0,0,…],…]
  * Images should be all generated images. E.g. (batch size, 3, 64, 64)
* Use make_grid(images) and save_image(images, path) (from torchvision.utils import save_image, make_grid) to save your image (8 images a row, 4 rows)
* The resolution of input for pretrained classifier is 64x64. You can degisn your own output resolution for generator and resize it.

#### Dataset
* Provided files
  * readme.txt, train.json, test.json, object.json, iclevr.zip, evaluator.py, classifier_weight.pth
* Iclver.zip
  * On the open source google drive lab6
* object.json
  * Dictionary of objects
  * 24 classes

![Dataset](/picture/Dataset.png "Dataset")

#### Output examples

![Output examples](/picture/Output_Example.png "Output examples")


#### Requirements
* Implement training, testing functions, and dataloader
* Choose your cGAN architecture
* Design your generator and discriminator
* Choose your loss functions
* Output the results based on test.json and new_test.json (will be released before demo)
