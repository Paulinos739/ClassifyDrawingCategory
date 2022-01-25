# Architectural Drawing Classification with Deep Learning

This Multi-Class Classification network, implemented with Keras, is used
here to develop a Computer Vision model capable to classify architectural drawings.

###Background
The aim of this project is to make the power of Computer Vision useful
to automatically sort and analyze the building-related data.
Here, a CNN is trained to distinguish between three types of
architectural drawings:
1. `elevation`
2. `floor-plan`
3. `section`

This project is part of an ongoing research from the [FID BAUdigital](https://kickoff.fid-bau.de/en/),
conducted at the Universitäts- und Landesbibliothek Darmstadt,
from the [TU Darmstadt](https://www.tu-darmstadt.de/index.en.jsp).

Due to copyright issues, the data used to train the models can not be publicly
shared in this repository.

###Usage

**multi_class_classification.py**

Data preparation was done through the flow_from_directory method. All images
have to sorted into different folders according to the categories.
The CNN uses softmax activation in the last layer and 3 neurons as output
to enable the [multi class classification](https://en.wikipedia.org/wiki/Multiclass_classification).
Useful Callbacks are installed, like tensorboard
or Learning Rate Scheduler.
Don't forget to adjust the learning rate in the **lr_time_based_scheduler**-function.
In this file a time-based decay is used.
Please play with the model's architecture, change the amount of layers or width,
to optimize performance.


**predict_classes.py**

After successful training of the model, use this file to run predictions on either
a single image sample or multiple images (both functions available).
This creates the drawings relevant metadata in a machine-readable format
The predictions will be converted to a list containing all information, using the pandas DataFrame method.
Available formats are: **.csv / .xlxs / .json**


### Queries
For questions or further ideas, email me at paul.arch@web.de