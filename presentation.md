# Building a Custom Object Detector with YOLOv5

# Why?

There are a multitude of out-of-the-box solutions for object detection hosted by FANG companies. Why are we interested in building our own?

1) Important or sensitive data remains on prem.
    - There are many situations where data cannot leave a specific network. Cloud solutions will obviously not work.
    - Own your data. Putting your data on a cloud AI platform for processing may not be agreeable to your users, or even you, depending on your use case. Sometimes, the value prop of your organization is dependent on sole ownership of important data.
2) More Deployment Options
    -  Send/Recv of Image and Video data requires sizeable bandwidth. Deploying your model straight to the source of the data can improve real time performance and lighten your network requirements.
3) Avoid Netwok Degredation
    - Critical and high uptime use cases may prefer to deploy their object detection model (or a backup) on networks not reliant on public DNS, internet, and regional risks.
4) Subscription or Usage Costs
5) Engineering Flexibility
    - Avoid vendor lock-in, data capture.

# Roadmap


# Data Requirements (YOLOv5)

## Label The Data

## Roboflow Universe Datasets

- 78 labeled images of T72's https://universe.roboflow.com/rigo-estrada/rut72
- 1,064 labeled images of vehicles https://universe.roboflow.com/capstoneproject/russian-military-annotated
- 1,028 labeled images of tanks https://universe.roboflow.com/rytjd0750-gmail-com/tankdetection/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true
- 6k labeled images of 5 vehicle types https://universe.roboflow.com/test-iabwn/tanks-9bk0z/images/80UQRTOnkhKSItTS2mpP?queryText=&pageSize=50&startingIndex=0&browseQuery=true

### Check your datasets

- Use the health check feature of roboflow
- Class Balance. You want an even distribution of your classes in order to prevent inherent model bias.
- Aspect Ratio

# Network Desciption - What is YOLOv5 doing?

While attempting to obtain the best data possible for your use-case, you are ultimately going to come to the conclusion that you need to understand the model itself. Having a real understanding of your neural network allows you to make educated guesses at what is good and bad data.

Unfortunately, understanding YOLOv5's network means briefly covering YOLOv1 through YOLOv4 as well. Luckily, the people who made the YOLOv1 through 4 algorithms published papers! YOLOv5 is an exception, but the open source community has generated some fantastic resources for that.

Timeline and Synopsis: https://www.v7labs.com/blog/yolo-object-detection

YOLO Paper: https://arxiv.org/pdf/1506.02640.pdf

YOLOv2: https://arxiv.org/abs/1612.08242

YOLOv3: https://arxiv.org/abs/1804.02767

YOLOv4: https://arxiv.org/abs/2004.10934 \
        - New authors \
        - Bag of Features



## YOLO Network Architecture

# Train

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

Demo of Twitter Monitor

Demo of Tank Detector in Squad

## CyberSecurity Use Cases - Lit Review

- **A Review of Computer Vision Methods in Network Security** at https://arxiv.org/abs/2005.03318
    - Anti-Phishing: Wang et al. [57] use recognized logos to check for valid domain ownership (They used Euclidean distance)
    - Anti-Phishing: Google Search-by-Image on the logo, check if top result is same owner (favicon)
    - Malware: Convert API Calls into Image Representation and train

