# 1. Face Recognition Demo For IS613
This repo is a simple demo using **plotly dash** for illustrating face recognition as well as potential vulnerabilities and one approach to countering it. Do note that this is plainly for demonstration purposes and hence there are certain areas that can be optimised better or refactored more for easier configuration.

Do note that a webcam is required for this demo!

# 2. Setting up environment
Create a new virtual environment (either conda or virtualenv) and run the following code to install the necessary packages:

```
pip install -r requirements.txt
```

# 3. Creating the database of images
A database will be created and/or populated automatically if not present during the execution of the web apps. However, before running it, please upload all face images into the folder called **data**. The image should be named as the person's name in the image itself (e.g. if there is an person named Tom, his image should be named as Tom.jpg inside the data directory). Lastly,

# 4. Running the demo
There are 2 demos present namely **demo.py** and **demo_anti_spoof.py**. You can either run one of them or run both at the same time. 

To run **demo.py**, simply type this in the command line with your virtual environment:

```
python demo.py
```
**demo.py** will be available on **localhost:8050**

To run **demo_anti_spoof.py**, simply type this in the command line with your virtual environment:

```
python demo_anti_spoof.py
```
**demo_anti_spoof.py** will be available on **localhost:8040**

To be able to navigate to either page at once, run both on separate command line and inside the navbar within the page there are ways to navigate to it.

# 5. Other Information
Do note that this is primarily ran on HaarCascades algorithm as well as HoG w SVM. Details on the algorithm will not be indicated here but can be easily found online.

Other than that, have fun and enjoy the demo! 

