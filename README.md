# Object-detection-project
Object detection on the surface with any video input

## Installation
    
    pip3 install -r requirements.txt

## Run

    python3 main.py 

## Parameters

* You can change parameters in the main.py file. 
  * Program can accept input from image, video and camera stream
  * Show options defined by **what_to_show** parameter:
    * Box around the object
    * Contours around the object
  * `detect_colored_objects` function can detect object of curtain color, but you would have to play with hue_koef parameter