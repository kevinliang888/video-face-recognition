# Video Face Recognition

* Generated datasets by extracting faces detected from six videos, using face\_recognition
* Trained the name classifier and gender classifier by Smaller VGGNet, got more than 98\% test accuracy.
* Captured the frames in the video, detected and classified name and gender of faces in each frame.
* Detected shot by comparing the histogram of consecutive frames, got better result than SIFT matching.
* Tracked movement of each face by implementing Center Tracker with intelligent functionality.

## Instruction
1. Download the datasets and models [here](https://github.com/kevinliang888/video-face-recognition/releases). Extract the zip file in the same directory. Datasets include gender dataset and character dataset which are used for training. Models include gender model and character model which are already trained. Videos include all the videos that I used for generating the datasets. You can jump to step 4 if you download the models and don't want to spend time on training.
2. Generate character datasets by running main in create_dataset.py
3. Train the model by running main in train.py. Here I use smaller VGGNet (smaller, more compact variant of the VGGNet network, introduced by Simonyan and Zisserman in their 2014 paper, Very Deep Convolutional Networks for Large Scale Image Recognition.)
4. Create output video by running video_detect.py. See bigbang_output.avi in the release.

## Display
![ezgif com-video-to-gif](https://user-images.githubusercontent.com/41521216/72198709-59884000-33ff-11ea-8766-e9c8eb175359.gif)

![ezgif com-video-to-gif (2)](https://user-images.githubusercontent.com/41521216/72198813-d2d46280-3400-11ea-87b8-d26ffd24d71f.gif)
