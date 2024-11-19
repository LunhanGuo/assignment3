#from jetson_inference import detectNet
#from jetson_utils import videoSource, videoOutput
import jetson.inference
import jetson.utils
import os
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

input_image_path = "/home/nvidia/jetson-inference/python/training/classification/data/train/cat/cat.0.jpg"

output_image_path = "/home/nvidia/jetson-inference/examples/cat.0.jpg"

img = jetson.utils.loadImage(input_image_path)

detections = net.Detect(img)
print("Detection Results:")
for detection in detections:
     print(f"ClassID:{detection.ClassID}")
     print("Confidence:{:.2f}".format(detection.Confidence))
     print("Left:{:.2f}".format(detection.Left))
     print("Top:{:2f}".format(detection.Top))
     print("Right:{:.2f}".format(detection.Right))
     print("Bottom:{:.2f}".format(detection.Bottom))
     print("Width:{:.2f}".format(detection.Width))
     print("Height:{:.2f}".format(detection.Height))
     print("Area:{:.2f}".format(detection.Area))
     print("Center:({:.2f},{:.2f})".format(detection.Center[0],detection.Center[1]))
     print("\n")


jetson.utils.saveImage(output_image_path,img)
   
print(f"Detection image saved to {output_image_path}")

