# lzh_fall_detection
part of code for paper fall detection using motion trajectory
The code data comes from the estimation results of openpose18 key points. 
The attitude estimation part refers to https://github.com/CMU-Perceptual-Computing-Lab/openpose, 
we use interpolation method to supplement and improve the joint points, and uses lstm to detect fall. 
The training part of the lstm model in this code can be run directly, and other functions have not been sorted out into a complete project
