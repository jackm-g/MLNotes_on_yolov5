# Building a Custom Object Detector with YOLOv5

Before we start, if you want to understand object detection from the fundamentals, this youtube playlist is the best resource I have been able to find: https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF


# Why?

There are a multitude of out-of-the-box solutions for object detection hosted by FANG companies. Why are we interested in building our own?

1) Important or sensitive data remains on prem.
    - There are many situations where data cannot leave a specific network. Cloud solutions will obviously not work.
    - Own your data. Putting your data on a cloud AI platform for processing may not be agreeable to your users, or even you, depending on your use case. Sometimes, the value prop of your organization is dependent on sole ownership of important data.
        - Medical data, financial analysis, or just generic PII.
2) More Deployment Options
    -  Send/Recv of Image and Video data requires sizeable bandwidth. Deploying your model straight to the source of the data can improve real time performance and lighten your network requirements.
    - Streaming video for real-time detection benefits tremendously from local deployments. Otherwise, your detection times may be bottlenecked by network latency or upload speed.
3) Avoid Netwok Degradation
    - Critical and high uptime use cases may prefer to deploy their object detection model (or a backup) on networks not reliant on public DNS, internet, and regional risks.
4) Subscription or Usage Costs
    - Cloud object detection can get expensive for users with tighter budgets. The price will heavily depend on your use case. For discrete instances requiring detection, cloud options are very viable and may even be free. Continuous detection may incur fairly high usage fees.
    - Google Cloud Vision on Images: about $7.50 for 5300 images. https://cloud.google.com/vision/pricing
    - Google Video Intelligence: $4,280/Mo for Person-Detection on a 24/7 Video Stream
    - With object detection, data transmission and storage costs may be high due to how large most video and image files are.
5) Engineering Flexibility
    - Avoid vendor lock-in, data capture.
    - Fully control the hardware and software stack. Cloud deployments may de-allocate your GPU's unexpectedly, or even have the wrong Hardware Driver versions for the libraries you need to make use of.
    - Not reliant on the API's exposed by the Cloud provider. This can increase your options for training the model if you need it done on a particularly niche use case.

# Roadmap
Before we begin learning to train a custom object detector, let's take a look at the steps.
# Data Requirements (YOLOv5)

## What is our input?
For Yolov5, a single unit of input consists of an image, `.jpg` or `.png`, and a label file, `.txt`. The image and label file should have the exact same name. This is important because the label file tells YOLO where the bounding boxes around each object in the image are. The label file should be formated in the YOLOv5 format, which is defined here.

**YOLOv5 Label Format** (Ref: https://roboflow.com/formats/yolov5-pytorch-txt)
Each label file contains rows that define the bounding boxes. They are of the format:\ `class_id center_x center_y width height`

An example would be:
```
1 0.617 0.3594 0.114 0.1738
2 0.094 0.3862 0.156 0.2360
```

It is worth noting that these values are normalized. This means the `center_x` refers to a point between 0 and 1, where 0 is the first pixel, X_MIN, and 1 is the last pixel on the axis, X_MAX. The same goes for the other values and their respective dimension. Normalizing allows us to keep the bounding boxes independent of image dimensions.

## YOLOv5 Dataset Selection

Dataset selection can be a chicken or the egg type problem when working with a model you aren't familiar with. You want to find the best data to train your model, but you don't often know what that dataset should be until you work with the model a bit. However, YOLO is a well-tread path and the authors of the YOLOv5 repository provide a robust requirements list. Before we even make our first "get me picture of X" google-search, let's look at the requirements.

YOLOv5 Data Tips: https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results

So ideally, we have 10k labelled instances of each class accross 1.5k images per class. The images should be "scenes" that reflect the operational environment the detection model will be deployed in. This means we want to use scenes with similar lighting, layout, and distance if possible.

Labels follow the garbage-in-garbage-out rule of machine learning. This means we want as many carefully done, error and gap free annotations as possible. Failing to label an object we want to detect is detrimental to our results.

Finally, including 0-10% background images that contain no labelled objects is a best practice. According to the documentation, this reduces false positives.

## Put Data on Local Filesystem

Now we have a good idea of what our data requirements are. We need to get a dataset that we can store locally for training. In this case, that means we need thousands of labelled images in a directory.

One option is to leverage Google's Custom Search API to get some images. (https://console.cloud.google.com/apis/library/customsearch.googleapis.com)

I used a python library called Google-Images-Seach v1.4.6. (https://pypi.org/project/Google-Images-Search/).

You will need to set up some API calls and there is pricing involved. Please refer to this directory for more info: TODO: INSERT_GITHUB_LINK

## Label The Data

Now that we have images in a directory, we want to label them. For this, we will head to https://www.makesense.ai/ -- this is a free browser tool that let's us annotate our images in the yolov5 format.

## Roboflow Universe Datasets

Labelling images can be time consuming. Assuming we can label an image in 5 seconds, we are going to need 1.3 hrs for every 1,000 images. There are a number of outsourced ways (Amazon Mechanical Turk) to get your images labelled, but we want to do this without external dependencies like that.

Roboflow Universe hosts a number of already labelled datasets, since it publically hosts any free-tier user's annotations and images. In my opinion, you will have more luck here than on Kaggle. In addition, these datasets are often labelled for YOLO, so you can expect to finda datasets tailored to our object detection use case.

Note: There is a very good chance someone has done/tried to detect what you are trying to detect before. Because transfer learning is a thing, there is a very good chance you can train your model on someone else's data before adding your own. We are going to do this.

**Online Data Sources**
- https://universe.roboflow.com/
- https://www.kaggle.com/datasets
- https://paperswithcode.com/datasets

**Data for our use case**

- 78 labeled images of T72's https://universe.roboflow.com/rigo-estrada/rut72
- 1,064 labeled images of vehicles https://universe.roboflow.com/capstoneproject/russian-military-annotated
- 1,028 labeled images of tanks https://universe.roboflow.com/rytjd0750-gmail-com/tankdetection/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true
- 6k labeled images of 5 vehicle types https://universe.roboflow.com/test-iabwn/tanks-9bk0z/images/80UQRTOnkhKSItTS2mpP?queryText=&pageSize=50&startingIndex=0&browseQuery=true

## Check your datasets

- Use the health check feature of roboflow
- Class Balance. You want an even distribution of your classes in order to prevent inherent model bias.
- Aspect Ratio
- Label Accuracy
- Do the pictures look similar to your use case?

NOTE: When downloading a dataset from Roboflow, it looks like they do a few data augmentations and throw those in to increase the size of your dataset, bundling that as part of the zip file you get.

# Network Desciption - What is YOLOv5 doing?

While attempting to obtain the best data possible for your use-case, you are going to come to the conclusion that you need to understand the model itself. Having a real understanding of your neural network allows you to make educated guesses at what is good and bad data.

In general, I think you will need to understand the following concepts:
- Intersect over Union
- Convolutions
- Max Pooling
- Nonmax suppression (https://www.youtube.com/watch?v=VAo84c1hQX8)
- Anchor boxes

## High-Level Description

Let's first just describe the network at a very high level, end-to-end. There are three primary components of the YOLO network: the Backbone, the Neck, and the Head.

Ref: https://blog.roboflow.com/yolov5-improvements-and-evaluation/

**Backbone:** A convolutional neural network that detects specific features such as edges, lines, curves etc. The size of each feature can vary. YOLOv5 uses a CNN called CSPDarknet53.

**Neck:** The neck combines features before passing them to the head. YOLOv5 uses Cross Stage Partial-PAN.
- CSP Paper: https://arxiv.org/pdf/1911.11929.pdf

**Head:** The head performs object localization, guessing where the bounding boxes are, and classification, guessing what type of object is in the bounding box. YOLOv5 uses the "YOLOv3 head".

## Full Description

Unfortunately, understanding YOLOv5's network means briefly covering YOLOv1 through YOLOv4 as well. Luckily, the people who made the YOLOv1 through 4 algorithms published papers! YOLOv5 is an exception, but the open source community has generated some fantastic resources for that.

Timeline and Synopsis: https://www.v7labs.com/blog/yolo-object-detection

### YOLO Paper: https://arxiv.org/pdf/1506.02640.pdf

### YOLOv2: https://arxiv.org/abs/1612.08242

### YOLOv3: https://arxiv.org/abs/1804.02767\
- Anchor boxes: YOLOv3 creates anchor boxes during training using the labelled bounding boxes in a custom dataset. K-means and genetic learning algorithms are used to do so.\
- Roboflow provides a handy step-by-step breakdown of how they are used, Ref: https://blog.roboflow.com/what-is-an-anchor-box/
```
1. Form thousands of candidate anchor boxes around the image
2. For each anchor box predict some offset from that box as a candidate box
3. Calculate a loss function based on the ground truth example
4. Calculate a probability that a given offset box overlaps with a real object
5. If that probability is greater than 0.5, factor the prediction into the loss function
6. By rewarding and penalizing predicted boxes slowly pull the model towards only localizing true objects
```
- Luckily for us, Ultralytic's YOLOv5 does not transfer the learned anchor boxes from the pre-training on MS COCO dataset. Rather, it trains on new anchor boxes using k-means and other methods on your custom dataset.

### YOLOv4: https://arxiv.org/abs/2004.10934 \
- New authors
- Bag of Features

### Cross Stage Partial (CSP) Networks Backbone
### SPPF
### PANet


## YOLO Network Architecture

The below github link has a complete graph of the YOLOv5 network, and illustrations of the various dataaugmentations YOLOv5 does to improve training.

Ref: https://github.com/ultralytics/yolov5/issues/6998

# Train

`run_train.sh`

I initially had CUDA out of memory errors for batch sizes > 8, even though train.py was reporting only 2.5GB of GPU_mem usage. With `export PIN_MEMORY='False'` I was able to increase batch size.

# The NVIDIA CUDA "Stack"
NVIDIA CUDA allows us to train a model like YOLOv5 on our at-home graphics card. As of now (November 2022), more and more state-of-the-art machine learning models can be run on the premium desktop hardware configurations owned by individual users.

## The Stack:

[Insert Image Here]

**Hardware:**\
    - GPU: NVIDIA CUDA Compatible GPU\
    - RAM: You will need a reasonable amount of RAM. At minimum the same size as your VRAM must be free.\

**Drivers**
    - Latest NVIDIA GPU Drivers\
        - These usually come with CUDA. (i.e The "Game-Ready Driver")

**System Libraries**
CUDNN\
    - Provides a set of primitives and functions commonly used in neural network based machine learning applications.\

CUDA-Toolkit

**Python Libraries**\
Pytorch + requirements.txt from Yolov5


**Application**\
Yolov5




# Deploy and Detect

## Demo of Twitter Monitor

Our first iteration of training used the ~1k images from rytjd0750's Roboflow Universe dataset. Performance was okay, but it did not always detect tanks, and kept labelling the sides of the twitter interface as tanks.

I then ran 200 epochs with 120 additional screenshots of twitter with tanks in them as part of the dataset. The idea was that the images of the twitter UI provides "context" that YOLO can learn from. I also zoomed in as much as I could on the webpage to make the feed's images larger.



Demo of Tank Detector in Squad

## CyberSecurity Use Cases - Lit Review

- **A Review of Computer Vision Methods in Network Security** at https://arxiv.org/abs/2005.03318
    - Anti-Phishing: Wang et al. [57] use recognized logos to check for valid domain ownership (They used Euclidean distance)
    - Anti-Phishing: Google Search-by-Image on the logo, check if top result is same owner (favicon)
    - Malware: Convert API Calls into Image Representation and train

