<div align="center">
    <img src="docs/NYC_MOCT.png" alt="NYC MOCT Logos" width="200">
    <img src="docs/nyc-dot-logo.png" alt="NYC DOT Logo" width="200">
    <img src="docs/NYU-Emblem.png" alt="NYU CUSP logo" width="200">
</div>

## Abstract

Data on pedestrian traffic flows and counts can be extremely useful for city planning. Which is why NYC has held a bi-annual pedestrian count in 114 key locations around the city. However this method is very limited, time consuming and expensive. The use of low-power, cost-effective AI chips such as Google’s Coral Edge TPU can reduce cost and flexibility of pedestrian counting. This capstone project will focus on training a pedestrian counting algorithm from preexisting labeled pedestrian datasets using TensorFlow. The trained TensorFlow model will be converted to a TensorFlow Lite model to be deployed on the Coral Edge TPU.  

## Goal

Many City agencies are involved in the use, planning, and design of public space but good data on pedestrian flows can be hard to come by. Manual counts take coordination and planning as well as staff costs. Computer-vision (CV) counting technologies are being tested in the city now but it is already clear that the infrastructure requirements (tapping into electricity and mounting to light poles) will be a limiting factor in using this technology more broadly and particularly for shorter-term studies where the infrastructure investment is not worth the time and effort. A low-cost, battery-powered CV sensor can help fill the data gap and allow agencies to utilize privacy-protected automated counts in short-term deployments with minimal infrastructure requirements.
In recent years, many hardware manufacturers have created development boards that support low-power computer vision (LPCV) applications. In addition, there has also been a fair amount of research done within academia to create low-power models for LPCV. This proposal aims to take advantage of recent technology advances to develop a hardware device that can be battery operated and utilized by New York City agencies to count pedestrians as they move through public space in the city. 
The NYC Department of Transportation Pedestrian Unit currently leads the pedestrian counting program in NYC. Most of the available pedestrian count data is manually counted from videos collected by contractors hired by the DOT. Building a low-power pedestrian counting device that is inexpensive, scalable, and accurate would help cut major costs for this important program. It would also grant horizontal flexibility in deployment at various locations in addition to vertical flexibility in scaling the program. As will be discussed, this work observes a tradeoff between computational efficiency and accuracy. Constructing a device that is efficient and low-power remains the most significant feature of this work. 

## Background

Good data on pedestrian flows can be some of the most difficult yet most important information streams about a city. Computer vision (CV) counting techniques have been tested as a means of automating this data stream. However, there proves to be critical economic and infrastructural limitations in deploying widespread CV counting technologies across a city. Recent developments in hardware have enabled the utilization of low-power computer vision (LPCV) on more mobile devices[^1]. This project aims to construct a low-power computer vision pedestrian counting device that is battery powered, long-lasting (2-4 weeks), and scalable.

In the past few years we have witnessed rapid and widespread shifting of our urban spaces due in large part to an ongoing environmental and epidemiological revolution. Recent literature on computer-vision pedestrian counting has sought to address some of these challenges by exploring mechanisms to automate social distancing requirements, for example.[^8] As cities seek to harden their infrastructure via technology, computer vision pedestrian counting has been used to address potential safety concerns.[^3],[^4],[^7]  Researchers have explored the use of computer vision as a means of pedestrian re-identification over varying distances and periods of time.[^3],[^4],[^7]  Computer vision has also been used to create neural networks as a means of pedestrian detection and counting for a variety of purposes.[^6],[^7] Generally there has been a wide scope of applications of computer vision for pedestrian counting. Some researchers have created labeled datasets through the use of multi-camera joint systems to attain more accurate measurements of pedestrian flow.[^2],[^3] Others have explored pedestrian counting and labeling in different environments and settings, like nighttime and crowded areas.[^5],[^6]

Identification of pre-labelled pedestrian datasets was critical in the early stages of this project for training and testing. The P-DESTRE dataset developed by Kumar et al. (2020) uses Unmanned Aerial Vehicles (UAVs) in a controlled outdoor environment for identification and reidentification of pedestrians.[^4] The Oxford Town Centre dataset is a widely cited dataset spanning over 10 years of implementation in pedestrian identification.[^8],[^11],[^12] It is sourced from a static CCTV in a busy pedestrian friendly plaza. It has 4501 annotated frames, which makes it ideal for training and testing.  

Individual privacy remains a large concern when identification is inherent to the practice. Several papers have addressed privacy concerns by training a model on a video game, Grand Theft Auto V, to fair success.[^3],[^10] Others have deployed schemes to maintain pedestrian privacy directly into their identification model.[^9] Concerns about data privacy anonymization further warrant the use of pre-labeled datasets (either gathered with consent or already anonymized). By conducting inference on the Edge via the Google Coral Dev Board we maintain privacy as machine learning and counting occurs on the device alone. 

For our pedestrian detection algorithm we elected to implement the You Only Look Once (YOLO) algorithm established by Redmon et al. (2016).[^14] The paper that established this methodology, model, and convolutional neural network (CNN) ultimately set forth the commonplace practice of solving the object detection problem as a regression problem predicting spatial bounding boxes with a given class property. There has been a great deal of development into this algorithm specifically. For example, Jee (2021) compared YOLOv3 to YOLOv3-Tiny in pedestrian identification on the Oxford Town Center dataset. The TinyML model was found to be preferable with slightly less accuracy but much greater speed.[^12] This serves as grounded and promising evidence for the use of the YOLO in a TinyML application. This rapidly changing field has seen recent developments in accuracy and speed with YOLOv7.[^13] For this project we elected to use the YOLOv5 framework due to the availability of trained PyTorch and Edge TPU models. 

For a large portion of this project we elected to implement and modify an existing pedestrian identification repository from Mikel Broström titled Real-time multi-camera multi-object tracker using YOLOb5 and StrongSORT with OSNet[^Mikel]. This well-documented repository was fairly simple to implement locally and on the Edge. However, the DeepSORT and later StrongSORT algorithm used for object tracking required an additional CNN. While this improved accuracy of the tracker, the additional neural network and model weights made the overall performance computationally expensive and power intensive. Our use case requires low-power implementation for scalability. Thus, we implemented a variation of the Norfair library from Tryolabs[^Norfair], which used a regular SORT algorithm for object tracking. The SORT algorithm (or Simple and Online Real Time Tracking) was first implemented by Bewley et al. (2017). By implementing a kalman filter and Hungarian Algorithm, SORT achieved comparable accuracy to state of the art models at 20x the speed.[^15] Tracking pedestrians is fundamental to counting them in order to identify unique objects as they enter and exit. 

The weights and class identities we utilized in the Norfair framework are trained on the COCO dataset established by Lin et al. (2014). The COCO dataset contains 91 object classes in 328,000 images. 

We were able to successfully implement the Norfair framework with the YOLOv5 model weights locally and on the Edge via the Google Coral Dev Board. By building a low-power and remote pedestrian counting device, we provide the City with a scalable and efficient means of tracking pedestrians at various intersections.

## Next Steps

TBD

## Deployment Instructions

### Install:

We recommend creating a new conda environment before installation and deployment
```bash
conda create --name LPCV
conda activate LPCV
```

Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a
[**Python>=3.6.0**](https://www.python.org/) environment

```bash
git clone https://github.com/Puturbold/LPCV-NYU-CAPSTONE.git  # clone
cd LPCV-NYU-CAPSTONE
pip install -r requirements.txt  # install
```

### How to run:  
```sh
python yolov5pedestrian.py --classes 0 --video video_file_path
```

| options| description|
|----|----|
|Coordinate Checker Arguments|
|`-i` or `--input`|Input image file name|
|`-s` or `--size`|Size of image (tool will create gray image with this size)|
|`-c` or `--cam`|ID number of webCam (camera). Starting from 0|
|`-sc` or `--scale`|Scale factor of display image. Display image size will be 1/2 when 2 is specified|
|Pedestrian Counter Arguments
|`--detector_path`|Yolov5 model path. Default is "yolov5m6.pt"|
|`--img_size`|YOLOv5 inference size (pixels). Default is "720"|
|`--conf_thres`|YOLOv5 object confidence threshold. Default is "0.25", pedestrian counter works best between 0.6-0.8|
|`--iou_thresh`|YOLOv5 IOU threshold for NMS. Default is "0.45", pedestrian counter works best at 0.6|
|`--classes`|Filter by class: --classes 0, or --classes 0 2 3, see COCO_classes.txt for full list|
|`--device`|Inference device: "cpu" or "cuda". Default is none, run default for edge deployment|
|`--track_points`|Track points: 'centroid' or 'bbox'. Default is "centroid"|
|`--video`|Put the video path - or 0 for camera. Default is 0, run default for edge deployment|
|`--init_delay`|Detection Initialization Delay -  must be less than hit_counter_max (15). Pedestrian counter works best at 15|

Following key commands only work for Coordinate checker UI
|key|description|
|----|----|
|`ESC`|Exit program|
|`p` or `<space>`|Pause / resume movie|

### Command line examples:  

```sh
python yolov5pedestrian.py --conf_thres 0.6 --iou_thresh 0.6 --classes 0 --video movie.mp4 --init_delay 15
python yolov5pedestrian.py --conf_thres 0.6 --iou_thresh 0.6 --classes 0 --video 0 --init_delay 15
python yolov5pedestrian.py --conf_thres 0.6 --iou_thresh 0.6 --classes 0 --device cuda --video movie.mp4 -init_delay 15
```

### Tuning and Testing Guide:

Tracked parameters log can be accessed by running mlflow server on local machine with following command and following locally hosted url

```sh
mlflow ui
```

Running yolov5pedestrian.py on CUDA device requires an cuda enable computer with required cuda software. Reference NVIDIA installation page https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html

### Video Cropping Guide:

After running yolov5pedestrian.py the coordinate checker window will pop-up. The user must identify the specific region of interest for the pedestrian counter by selecting two points on the video feed. The specific region of interest is dependent on the deployment scenario.

#### Notes on cropped image size and model input weights:
Accuracy of pedestrian counter can be effected by the input video resolution and cropped region of interest (ROI) dimensions. Current .tflite model weight takes input size 256x256, therefore the model will reshape all input images to fit 256x256. During deployment, user should be cautious of input images size that are too large or two small. 

## References

[^1]:Alyamkin, S., Ardi, M., Berg, A. C., Brighton, A., Chen, B., Chen, Y., ... & Zhuo, S. (2019). Low-power computer vision: Status, challenges, and opportunities. IEEE Journal on Emerging and Selected Topics in Circuits and Systems, 9(2), 411-421.

[^2]:Chavdarova, T., Baqué, P., Bouquet, S., Maksai, A., Jose, C., Bagautdinov, T., ... & Fleuret, F. (2018). Wildtrack: A multi-camera hd dataset for dense unscripted pedestrian detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 5030-5039).

[^3]:Kohl, P., Specker, A., Schumann, A., & Beyerer, J. (2020). The mta dataset for multi-target multi-camera pedestrian tracking by weighted distance aggregation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (pp. 1042-1043).

[^4]:Kumar, S. A., Yaghoubi, E., Das, A., Harish, B. S., & Proença, H. (2020). The p-destre: A fully annotated dataset for pedestrian detection, tracking, and short/long-term re-identification from aerial devices. IEEE Transactions on Information Forensics and Security, 16, 1696-1708.

[^5]:Jia, X., Zhu, C., Li, M., Tang, W., & Zhou, W. (2021). LLVIP: A Visible-infrared Paired Dataset for Low-light Vision. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 3496-3504).

[^6]:Zhang, L., Shi, M., & Chen, Q. (2018, March). Crowd counting via scale-adaptive convolutional neural network. In 2018 IEEE Winter Conference on Applications of Computer Vision (WACV) (pp. 1113-1121). IEEE.

[^7]:Li, W., Zhao, R., Xiao, T., & Wang, X. (2014). Deepreid: Deep filter pairing neural network for person re-identification. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 152-159).

[^8]:Yang, D., Yurtsever, E., Renganathan, V., Redmill, K. A., & Özgüner, Ü. (2021). A vision-based social distancing and critical density detection system for COVID-19. Sensors, 21(13), 4608.

[^9]:Yang, H., Zhou, Q., Ni, J., Li, H., & Shen, X. (2020). Accurate image-based pedestrian detection with privacy preservation. IEEE Transactions on Vehicular Technology, 69(12), 14494-14509.

[^10]:Fabbri, M., Brasó, G., Maugeri, G., Cetintas, O., Gasparini, R., Ošep, A., ... & Cucchiara, R. (2021). MOTSynth: How Can Synthetic Data Help Pedestrian Detection and Tracking?. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 10849-10859).

[^11]:Benfold, B., & Reid, I. (2011, June). Stable multi-target tracking in real-time surveillance video. In CVPR 2011 (pp. 3457-3464). IEEE.

[^12]:Jee, C. Y. (2021). Social Distancing Detector for Pedestrians Using Deep Learning Algorithms (Doctoral dissertation, Tunku Abdul Rahman University College).

[^13]:Wang, C. Y., Bochkovskiy, A., & Liao, H. Y. M. (2022). YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors. arXiv preprint arXiv:2207.02696.

[^14]:Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 779-788).

[^15]:Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016, September). Simple online and realtime tracking. In 2016 IEEE international conference on image processing (ICIP) (pp. 3464-3468). IEEE.

[^16]:Lin, T. Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., ... & Zitnick, C. L. (2014, September). Microsoft coco: Common objects in context. In European conference on computer vision (pp. 740-755). Springer, Cham.

[^Mikel]:Mikel Broström (2022). Real-time multi-camera multi-object tracker using YOLOv5 and StrongSORT with OSNet. https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet

[^Norfair]:Joaquín Alori, Alan Descoins, KotaYuhara, David, facundo-lezama, Braulio Ríos, fatih, shafu.eth, Agustín Castro, & David Huh. (2022). tryolabs/norfair: v1.0.0 (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.6596178 (https://github.com/tryolabs/norfair)