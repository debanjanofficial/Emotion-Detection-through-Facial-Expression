# Emotion Detection through FacialExpression

## INTRODUCTION
“2018 is the year when machines learn to grasp human emotions” -Andrew
Moore, the dean of computer science at Carnegie Mellon.
With the advent of modern technology our desires went high and it binds no bounds. In the present
era a huge research work is going on in the field of digital image and image processing. The way of
progression has been exponential and it is ever increasing. Image Processing is a vast area of
research in present day world and its applications are very widespread. Image processing is the field
of signal processing where both the input and output signals are images. One of the most important
application of Image processing is Facial expression recognition. Our emotion is revealed by the
expressions in our face. Facial Expressions plays an important role in interpersonal communication.
Facial expression is a non-verbal scientific gesture which gets expressed in our face as per our
emotions. Automatic recognition of facial expression plays an important role in artificial intelligence
and robotics and thus it is a need of the generation. Some application related to this include
Personal identification and Access control, Videophone and Teleconferencing, Forensic application,
Human-Computer Interaction, Automated Surveillance,
Cosmetology and so on. The objective of this project is to develop Automatic Facial Expression Recognition
System which can take human facial images containing some expression as input and recognize and classify
it into seven different expression class such as:
#### ❖ Happy
#### ❖ Sad
#### ❖ Fear
#### ❖ Surprise
#### ❖ Neutral 
#### ❖ Disgust
#### ❖ Angry
<img width="528" alt="image" src="https://github.com/debanjanofficial/Emotion-Detection-through-Facial-Expression/assets/78428813/78ac0421-8684-4459-838b-0eb453a754d2">

Several Projects have already been done in this fields and our
goal will not only be to develop an Automatic Facial Expression Recognition System but also improving the
accuracy of this system compared to the other available systems.

## REQUIREMENTS
### ❖Software:
➢ Anaconda Navigator
➢ Pycharm
➢ Jupitar Notebook
➢ Anaconda Prompt
➢ Microsoft Word 2013
➢ Microsoft PowerPoint 2013
➢ Database Storage: Microsoft Excel 2013
### ❖Hardware:
➢ Processor: INTEL CORE i5 8th Gen with minimum 1.6 GHz speed
➢ RAM: Minimum 8 GB
➢ Hard Disk: Minimum 2 TB
### ❖Operating System:
➢ Windows 10
### ❖Language:
➢ Python 3.7
### ❖Library:
➢ NumPy
➢ Pandas
➢ Keras
➢ OpenCV
➢ TensorFlow


## PLANNING
The steps we followed while developing this project are:
#### ➢ Analysis of the problem statement.
#### ➢ Gathering of the requirement specification ➢ Analyzation of the feasibility of the project.
#### ➢ Development of a general layout.
#### ➢ Going by the journals regarding the previous related works on this field.
#### ➢ Choosing the method for developing the algorithm.
#### ➢ Analyzing the various pros and cons.
#### ➢ Starting the development of the project
#### ➢ Installation of software like ANACONDA.
#### ➢ Developing an algorithm.
#### ➢ Analyzation of algorithm by guide.
#### ➢ Coding as per the developed algorithm in PYTHON. 

## DESIGN
### Data Flow Program
A data flow diagram (DFD) is a graphical representation of the "flow" of data
through an information system, modelling its process aspects. A DFD is often
used as a preliminary step to create an overview of the system without going
into great detail, which can later be elaborated. DFDs can also be used for the
visualization of data processing (structured design).
A DFD shows what kind of information will be input to and output from the
system, how the data will advance through the system, and where the data will
be stored. It does not show information about process timing or whether
processes will operate in sequence or in parallel, unlike a traditional structured
flowchart which focuses on control flow, or a UML activity workflow diagram,
which presents both control and data flows as a unified model.
Data flow diagrams are also known as bubble charts. DFD is a designing tool used
in the top down approach to Systems Design.
### Symbols & Notations used in DFDs:
Using any convention’s DFD rules or guidelines, the symbols depict the four
components of data flow diagrams –
#### External entity: 
an outside system that sends or receives data,
communicating with the system being diagrammed. They are the sources and
destinations of information entering or leaving the system. They might be an
outside organization or person, a computer system or a business system. They
are also known as terminators, sources and sinks or actors. They are typically
drawn on the edges of the diagram.
#### Process: 
any process that changes the data, producing an output. It might
perform computations, or sort data based on logic, or direct the data flow based
on business rules. 
#### Data Store: 
files or repositories that hold information for later use, such as a
database table or a membership form. 
#### Data Flow: 
the route that data takes between the external entities, processes
and data stores. It portrays the interface between the other components and is
shown with arrows, typically labeled with a short data name, like “Billing
details.”

## DFD Levels and Layers:
A data flow diagram can dive into progressively more detail by using levels and
layers, zeroing in on a particular piece. DFD levels are numbered 0, 1 or 2, and
occasionally go to even Level 3 or beyond. The necessary level of detail depends
on the scope of what you are trying to accomplish.
#### DFD Level 0 
It is also called a Context Diagram. It’s a basic overview of the whole
system or process being analyzed or modeled. It’s designed to be an at-a-glance
view, showing the system as a single high-level process, with its relationship to
external entities. It should be easily understood by a wide audience, including
stakeholders, business analysts, data analysts and developers.
#### DFD Level 1 
This provides a more detailed breakout of pieces of the Context Level
Diagram. You will highlight the main functions carried out by the system, as
you break down the high-level process of the Context Diagram into its sub
processes.
#### DFD Level 2 
Then goes one step deeper into parts of Level 1. It may require
more text to reach the necessary level of detail about the system’s functioning. 

Progression to Levels 3, 4 and beyond is possible, but going beyond Level 3 is
uncommon. Doing so can create complexity that makes it difficult to communicate,
compare or model effectively.
Using DFD layers, the cascading levels can be nested directly in the diagram,
providing a cleaner look with easy access to the deeper dive. 


## ALGORITHM
##### Step 1: Collection of a data set of images. (In this case we are using FER2013 database of 35887 pre-cropped, 48-by-48-pixel grayscale images of faces each labeled with one of the 7 emotion classes: anger, disgust, fear, happiness, sadness, surprise, and neutral.
##### Step 2: Pre-processing of images.
##### Step 3: Detection of a face from each image.
##### Step 4: The cropped face is converted into grayscale images.
##### Step 5: The pipeline ensures every image can be fed into the input layer as a (1, 48, 48) NumPy array.
##### Step 6: The NumPy array gets passed into the Convolution2D layer.
##### Step 7: Convolution generates feature maps.
##### Step 8: Pooling method called MaxPooling2D that uses (2, 2) windows across the feature map only keeping the maximum pixel value.
##### Step 9: During training, Neural network Forward propagation and Backward propagation performed on the pixel values.
##### Step 10: The Softmax function presents itself as a probability for each emotion class. The model is able to show the detail probability composition of the emotions in the face. 

## ANALYSIS
Computer vision (CV) is the field of study that helps computers to study using
different techniques and methods so that it can capture what exists in an image
or a video. There are a large number of applications of computer vision that
are present today like facial recognition, driverless cars, medical diagnostics,
etc. We will discuss one of the interesting applications of CV that is Emotion
Detection through facial expressions. CV can recognize and tell you what your
emotion is by just looking at your facial expressions. It can detect whether you
are angry, happy, sad, etc.
The article demonstrates a computer vision model that we will build using
Keras and VGG16 – a variant of Convolutional Neural Network. We will use
this model to check the emotions in real-time using OpenCV and webcam.
Here, we talk about
➢ Downloading the data
➢ Building the VGG Model for emotion detection
➢ Training of the VGG model efficiently so that it can recognize the
emotion
➢ Testing of the model in real-time using webcam
### The Dataset:
The name of the data set is fer2013 which is an open-source data set that
was made publicly available for a Kaggle competition. It contains 48 X
48-pixel grayscale images of the face. There are seven categories
(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
present in the data. The CSV file contains two columns that are emotion
that contains numeric code from 0-6 and a pixel column that includes a
string surrounded in quotes for each image.
### Implementing VGG16 Network for Classification of Emotionswith GPU:
First, we need to enable GPU in the Google Colab to get fast processing.
We can enable it by going to ‘Runtime’ in Google Colab and then clicking
on ‘Change runtime type’ and select GPU. Once it is enabled we will now
import the required libraries for building the network. The code is uploaded.

### VGG16 Model for Emotion Detection:
Now it’s time to design the CNN model for emotion detection with
different layers. We start with the initialization of the model followed
by batch normalization layer and then different convents layers with
ReLu as an activation function, max pool layers, and dropouts to do
learning efficiently. You can also change the architecture by initiating
the layers of your choices with different numbers of neurons and
activation functions. 

### Testing the Model in Real-Time using OenCV and WebCam:
Now we will test the model that we build for emotion detection in
realtime using OpenCV and webcam. To do so we will write a python
script. We will use the Jupyter notebook in our local system to make use
of a webcam. First, we will install a few libraries that are required. Code is uploaded in the testing.

### Emotion Detection Demo:
<img width="1175" alt="image" src="https://github.com/debanjanofficial/Emotion-Detection-through-Facial-Expression/assets/78428813/6c51db24-c77f-4494-8a41-6e630b95a1f6">

## CONCLUSION
In this case, when the model predicts incorrectly, the correct label is often
the second most likely emotion.
The facial expression recognition system presented in this research work
contributes a resilient face recognition model based on the mapping of behavioral
characteristics with the physiological biometric characteristics. The physiological
characteristics of the human face with relevance to various expressions such as
happiness, sadness, fear, anger, surprise and disgust are associated with
geometrical structures which restored as base matching template for the
recognition system.
The behavioral aspect of this system relates the attitude behind different
expressions as property base. The property bases are alienated as exposed and
hidden category in genetic algorithmic genes. The gene training set evaluates the
expressional uniqueness of individual faces and provide a resilient expressional
recognition model in the field of biometric security.
The design of a novel asymmetric cryptosystem based on biometrics having
features like hierarchical group security eliminates the use of passwords and smart
cards as opposed to earlier cryptosystems. It requires a special hardware support
like all other biometrics system. This research work promises a new direction of
research in the field of asymmetric biometric cryptosystems which is highly
desirable in order to get rid of passwords and smart cards completely.
Experimental analysis and study show that the hierarchical security structures are
effective in geometric shape identification for physiological traits. 

## FUTURE COURSE OF ACTION
It is important to note that there is no specific formula to build a neural network that would
guarantee to work well. Different problems would require different network architecture
and a lot of trail and errors to produce desirable validation accuracy. This is the reason
why neural nets are often perceived as "black box algorithms."
In this project we got an accuracy of almost 70% which is not bad at all comparing all the
previous models. 
But we need to improve in specific areas like-
➢ number and configuration of convolutional layers
➢ number and configuration of dense layers
➢ dropout percentage in dense layers
But due to lack of highly configured system we could not go deeper into dense neural
network as the system gets very slow and we will try to improve in these areas in future.
We would also like to train more databases into the system to make the model more and
more accurate but again resources becomes a hindrance in the path and we also need to
improve in several areas in future to resolve the errors and improve the accuracy.
Having examined techniques to cope with expression variation, in future it may be
investigated in more depth about the face classification problem and optimal fusion of
color and depth information. Further study can be laid down in the direction of allele of
gene matching to the geometric factors of the facial expressions. The genetic property
evolution framework for facial expressional system can be studied to suit the requirement
of different security models such as criminal detection, governmental confidential security
breaches etc.

## REFERENCES
#### “Real Time Emotion Analysis Using Keras” by Neha Yadav –
https://www.youtube.com/watch?v=DtBu1u5aBsc
#### Information about “Emotion Detection Through Facial Expression" from 
http://www.google.com/search
#### Dataset “fer2013” from
https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition
#### “Emotion Detection – Python project using Machine Learning & OpenCV” by Misbah Mohammed -
https://www.youtube.com/watch?v=AP9e4ny_KHc
