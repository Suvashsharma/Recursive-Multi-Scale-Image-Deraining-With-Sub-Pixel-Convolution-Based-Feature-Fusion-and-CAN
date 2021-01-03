# Recursive-Multi-Scale-Image-Deraining-With-Sub-Pixel-Convolution-Based-Feature-Fusion-and-Context Aggregation
In this paper, we propose a new single image deraining architecture with competitive deraining performance. Specifically we use a recursive technique of deraining with two modules cascaded each other. The first module -called front-end module- is dedicated to remove the rain at coarse level and is based on dense fusion of lower label features followed by sub-pixel convolutions. The second module -called refinement module- is dedicated to further remove the remnant of rain streaks and is based on context aggregation networks. The overall architecture is trained end to end and is capable to produce high-quality results -on both real-world and synthetic datasets- that are superior to several famous techniques in the literature.

<img src = "/Graphical_Abstract/Graphical_abstract.PNG" >

# Requirements:
  -Ubuntu >=16.04\
  -Python3.6\
  -Pytorch >=0.4\
  -opencv-python, tensorboardX
# Training:
  -Download the dataset Rain100H, Rain100L, Rain12, DDN dataset and put them in subsequent folder inside Datasets/train/ and /Datasets/test/.\
  -Set the data and model saving directories in the script "train.py".
  -CD to the master folder and run the command for training as given in commands.txt file (python3.6 train.py).\
# Testing:  
  - Put the test dataset in /Datasets/test/ directory.\
  - Run the command "python3.6 test.py".\ 
