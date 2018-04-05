from __future__ import division

from cStringIO import StringIO
from PIL import Image
import comm
import forwarder
import time
import uuid
import random
import struct
import os
import glob
import numpy as np





# training mode from the app is sending us a fully annotated image
def HandleTrainingImage(msg):
    # Note: the packet contains a null terminated string representing the file name
    # to be used to sve the image, followed by PNG image contents
    parts = msg.split('\x00', 1)
    
    fileName = parts[0]
    filePath = "../ocr/permanent/"+fileName
    
    print("saving " + filePath)
    print("size of image: " + str(len(parts[1])) )
    f = open(filePath, 'wb')
    f.write(parts[1])
    f.close()
    

comm.subscriber(comm.endpoint_sub_TrainingImages, HandleTrainingImage)


print("Begin server main loop...")
while True:
    didProcessMessage = comm.PollSockets()
    if didProcessMessage == False:
        time.sleep(0.001)



    