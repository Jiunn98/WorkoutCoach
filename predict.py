import cv2
import time
import numpy as np
import glob
import math
#from skimage.feature import greycomatrix, greycoprops
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle
import subprocess
import shlex
import json
from moviepy import *
import argparse

protoFile = "OpenPose/pose_deploy_linevec.prototxt"
weightsFile = "OpenPose/pose_iter_440000.caffemodel"
nPoints = 18
POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

inWidth = 368
inHeight = 368
threshold = 0.1

def cal_angle1(point1, point2, point3):
  if point2[0] - point1[0] != 0 and point3[0] - point2[0] != 0:
    s1 = (-point2[1] - -point1[1]) / (point2[0] - point1[0])
    s2 = (-point3[1] - -point2[1]) / (point3[0] - point2[0])

    ang = math.degrees(math.atan((s2-s1)/(1+(s2*s1))))
  
    if ang < 0:
      ang = 180 + ang

    return ang

  else:
    return 0

def cal_angle2(point1, point2, point3, point4):
  if point2[0] - point1[0] != 0 and point4[0] - point3[0] != 0:
    m1 = (-point2[1] - -point1[1]) / (point2[0] - point1[0])
    A = math.atan(m1) * 180 / 3.14159265

    m2 = (-point4[1] - -point3[1]) / (point4[0] - point3[0])
    B = math.atan(m2) * 180 / 3.14159265

    return int(A - B)
  else:
    return 0

def cal_shoulderpress(wrist, shoulder):
  if wrist[1] <= shoulder[1]:
    return shoulder[1] - wrist[1]
  else:
    return 0

def get_rotation(file_path_with_file_name):
    cmd = "ffprobe -loglevel error -select_streams v:0 -show_entries stream_tags=rotate -of default=nw=1:nk=1"
    args = shlex.split(cmd)
    args.append(file_path_with_file_name)
    # run the ffprobe process, decode stdout into utf-8 & convert to JSON
    ffprobe_output = subprocess.check_output(args).decode('utf-8')
    if len(ffprobe_output) > 0:  # Output of cmdis None if it should be 0
        ffprobe_output = json.loads(ffprobe_output)
        rotation = ffprobe_output

    else:
        rotation = 0

    return rotation

def bicepcurl(trainingdata):
  scaler = MinMaxScaler() #perform feature scaling
  scaler.fit(trainingdata)
  MinMaxScaler(copy=True, feature_range=(0, 1))
  scaler.transform(trainingdata)

  loadedknn = pickle.load(open('Model/bicep_curl.sav', "rb")) #change this
  value = loadedknn.predict(trainingdata)
  b1 = 0
  b2 = 0
  b3 = 0
  b4 = 0
  b5 = 0
  b6 = 0
  g1 = 0
  for x in value:
    if x == 'b1_bicepcurl':
      b1 += 1
    if x == 'b2_bicepcurl':
      b2 += 1
    if x == 'b3_bicepcurl':
      b3 += 1
    if x == 'b4_bicepcurl':
      b4 += 1
    if x == 'b5_bicepcurl':
      b5 += 1
    if x == 'b6_bicepcurl':
      b6 += 1
    if x == 'g1_bicepcurl':
      g1 += 1

  workout = {'B1 Bicep Curl': b1, 'B2 Bicep Curl': b2, 'B3 Bicep Curl': b3, 'B4 Bicep Curl': b4, 'B5 Bicep Curl': b5, 'B6 Bicep Curl': b6, 'G1 Bicep Curl': g1}
  #pos_result = max(workout, key=workout.get)
  #print("Posture Evaluation: ", max(workout, key=workout.get))
  if max(workout, key=workout.get) == 'B1 Bicep Curl':
    print("Detected Posture Characteristics: ")
    print("The middle of the body swings forward slightly, the lower back overarches and the legs probably dip a little. ")
    print("Improper Posture Category: Relying on Momentum and swinging motions")
    print("Confident Level: ", str((b1/(b1+b2+b3+b4+b5+b6+g1))*100), "%")
    print()
    print("Suggested Posture Collection: ")
    print("This might due to over heavy weight lifted, you should consider dropping some weight.")
    print("You also need to keep your back straight while performing the exercise")
    print("Swinging the weight to the back to gain Momentum is considered as cheat repetition, so don't do that!")
    #pos_con = str((b1/(b1+b2+b3+b4+b5+b6+g1))*100)
  if max(workout, key=workout.get) == 'B2 Bicep Curl':
    print("Detected Posture Characteristics: ")
    print("Posture seems fine, but the repetition speed is too fast to be considered as normal repetition.")
    print("Improper Posture Category: Rushing Through Reps")
    print("Confident Level: ", str((b2/(b1+b2+b3+b4+b5+b6+g1))*100), "%")
    print()
    print("Suggested Posture Collection: ")
    print("You may maintain the current posture you are performing, but slow down when you lowering the weight.")
    print("Rhis can make sure you are able to build more strength.")
    #pos_con = str((b2/(b1+b2+b3+b4+b5+b6+g1))*100)
  if max(workout, key=workout.get) == 'B3 Bicep Curl':
    print("Detected Posture Characteristics: ")
    print("You are not fully lifting the weight, this is only considered as half repetition.")
    print("Improper Posture Category: Lower Partial Range of Motion")
    print("Confident Level: ", str((b3/(b1+b2+b3+b4+b5+b6+g1))*100), "%")
    print()
    print("Suggested Posture Collection: ")
    print("In order to finish a full repetition, you need to make sure you hit full range of motion of your lower hand.")
    print("Lift the weight all the to the top by not moving your elbow.")
    #pos_con = str((b3/(b1+b2+b3+b4+b5+b6+g1))*100)
  if max(workout, key=workout.get) == 'B4 Bicep Curl':
    print("Detected Posture Characteristics: ")
    print("You are not fully lifting the weight, this is only considered as half repetition.")
    print("Improper Posture Category: Upper Partial Range of Motion")
    print("Confident Level: ", str((b4/(b1+b2+b3+b4+b5+b6+g1))*100), "%")
    print()
    print("Suggested Posture Collection: ")
    print("In order to finish a full repetition, you need to make sure you hit full range of motion of your lower hand.")
    print("Extend the elbow all the way to the bottom when you lowering the weight.")
    #pos_con = str((b4/(b1+b2+b3+b4+b5+b6+g1))*100)
  if max(workout, key=workout.get) == 'B5 Bicep Curl':
    print("Detected Posture Characteristics: ")
    print("Your elbows flare out or move behind your back as you lift the weight")
    print("Improper Posture Category: Moving Your Elbow")
    print("Confident Level: ", str((b5/(b1+b2+b3+b4+b5+b6+g1))*100), "%")
    print()
    print("Suggested Posture Collection: ")
    print("Keep your elbow still and stick your upper hand to your torso and you are good to go.")
    print("Usually this improper posture is due to excessive weight, therefore I encourage you to drop some weight.")
    #pos_con = str((b5/(b1+b2+b3+b4+b5+b6+g1))*100)
  if max(workout, key=workout.get) == 'B6 Bicep Curl':
    print("Detected Posture Characteristics: ")
    print("The middle of the body swings backward slightly to support the weight.")
    print("Improper Posture Category: Not Keeping Your Core Tight")
    print("Confident Level: ", str((b6/(b1+b2+b3+b4+b5+b6+g1))*100), "%")
    print()
    print("Suggested Posture Collection: ")
    print("Keep the back straight all the time when you lift the weight up.")
    #pos_con = str((b6/(b1+b2+b3+b4+b5+b6+g1))*100)
  if max(workout, key=workout.get) == 'G1 Bicep Curl':
    print("Detected Posture Characteristics: ")
    print("The back is straight, elbow is not moving and it is stick to the back, performing full range of motion.")
    print("Improper Posture Category: None!")
    print("Congratulations! You are performing the correct posture, keep it up!")
    print("Confident Level: ", str((g1/(b1+b2+b3+b4+b5+b6+g1))*100), "%")
    #pos_con = str((g1/(b1+b2+b3+b4+b5+b6+g1))*100)

  #return pos_result, pos_con

def frontraise(trainingdata):
  scaler = MinMaxScaler() #perform feature scaling
  scaler.fit(trainingdata)
  MinMaxScaler(copy=True, feature_range=(0, 1))
  scaler.transform(trainingdata)

  loadedknn = pickle.load(open('Model/front_raise.sav', "rb")) #change this
  value = loadedknn.predict(trainingdata)
  b1 = 0
  b2 = 0
  b3 = 0
  b4 = 0
  b5 = 0
  g1 = 0
  for x in value:
    if x == 'b1_frontraise':
      b1 += 1
    if x == 'b2_frontraise':
      b2 += 1
    if x == 'b3_frontraise':
      b3 += 1
    if x == 'b4_frontraise':
      b4 += 1
    if x == 'b5_frontraise':
      b5 += 1
    if x == 'g1_frontraise':
      g1 += 1

  workout = {'B1 Front Raise': b1, 'B2 Front Raise': b2, 'B3 Front Raise': b3, 'B4 Front Raise': b4, 'B5 Front Raise': b5, 'G1 Front Raise': g1}
  #pos_result = max(workout, key=workout.get)
  #print("KNN Classifier: ", max(workout, key=workout.get))
  if max(workout, key=workout.get) == 'B1 Front Raise':
    print("Detected Posture Characteristics: ")
    print("The torso is sawying backwards as you performing the repetition")
    print("Improper Posture Category: Rocking")
    print("Confident Level: ", str((b1/(b1+b2+b3+b4+b5+g1))*100), "%")
    print()
    print("Suggested Posture Collection: ")
    print("Keep the back straight all the time when you lift the weight up.")
    #pos_con = str((b1/(b1+b2+b3+b4+b5+g1))*100)
  if max(workout, key=workout.get) == 'B2 Front Raise':
    print("Detected Posture Characteristics: ")
    print("The middle of the body swings forward slightly, the lower back overarches and the legs probably dip a little, hands swing behind torso to gain momentum")
    print("Improper Posture Category: Using Momentum")
    print("Confident Level: ", str((b2/(b1+b2+b3+b4+b5+g1))*100), "%")
    print()
    print("Suggested Posture Collection: ")
    print("This might due to over heavy weight lifted, you should consider dropping some weight.")
    print("You also need to keep your back straight while performing the exercise")
    print("Swinging the weight to the back to gain Momentum is considered as cheat repetition, so don't do that!")
    #pos_con = str((b2/(b1+b2+b3+b4+b5+g1))*100)
  if max(workout, key=workout.get) == 'B3 Front Raise':
    print("Detected Posture Characteristics: ")
    print("You lift the weight up to unnecessary height, which is above the shoulders")
    print("Improper Posture Category: Lifting above Shoulders")
    print("Confident Level: ", str((b3/(b1+b2+b3+b4+b5+g1))*100), "%")
    print()
    print("Suggested Posture Collection: ")
    print("This might due to over light weight lifted, you should consider adding some weight.")
    print("Lift the weight up to same level with shoulders is enough.")
    print("Over lifting might increase the risk of injury to the surprisingly delicate shoulder joint.")
    #pos_con = str((b3/(b1+b2+b3+b4+b5+g1))*100)
  if max(workout, key=workout.get) == 'B4 Front Raise':
    print("Detected Posture Characteristics: ")
    print("You are not fully lifting the weight, this is only considered as half repetition.")
    print("Improper Posture Category: Partial Range of Motion")
    print("Confident Level: ", str((b4/(b1+b2+b3+b4+b5+g1))*100), "%")
    print()
    print("Suggested Posture Collection: ")
    print("This might due to over over weight lifted, you should consider dropping some weight.")
    print("You need to lift the weight all the way up to the same level of shoulders to consider as a full repettion.")
    #pos_con = str((b4/(b1+b2+b3+b4+b5+g1))*100)
  if max(workout, key=workout.get) == 'B5 Front Raise':
    print("Detected Posture Characteristics: ")
    print("Posture seems fine, but the repetition speed is too fast to be considered as normal repetition.")
    print("Improper Posture Category: Rushing Through Reps")
    print("Confident Level: ", str((b5/(b1+b2+b3+b4+b5+g1))*100), "%")
    print()
    print("Suggested Posture Collection: ")
    print("You may maintain the current posture you are performing, but slow down when you lowering the weight.")
    print("Rhis can make sure you are able to build more strength.")
    #pos_con = str((b5/(b1+b2+b3+b4+b5+g1))*100)
  if max(workout, key=workout.get) == 'G1 Front Raise':
    print("Detected Posture Characteristics: ")
    print("The back is straight, the hands are lifting the weight up with a slight bend in the elbows, performing full range of motion.")
    print("Improper Posture Category: None!")
    print("Congratulations! You are performing the correct posture, keep it up!")
    print("Confident Level: ", str((g1/(b1+b2+b3+b4+b5+g1))*100), "%")
    #pos_con = str((g1/(b1+b2+b3+b4+b5+g1))*100)

  #return pos_result, pos_con

def shoulderpress(trainingdata):
  scaler = MinMaxScaler() #perform feature scaling
  scaler.fit(trainingdata)
  MinMaxScaler(copy=True, feature_range=(0, 1))
  scaler.transform(trainingdata)

  loadedknn = pickle.load(open('Model/shoulder_press.sav', "rb")) #change this
  value = loadedknn.predict(trainingdata)
  b1 = 0
  b2 = 0
  b3 = 0
  b4 = 0
  b5 = 0
  g1 = 0
  for x in value:
    if x == 'b1_shoulderpress':
      b1 += 1
    if x == 'b2_shoulderpress':
      b2 += 1
    if x == 'b3_shoulderpress':
      b3 += 1
    if x == 'b4_shoulderpress':
      b4 += 1
    if x == 'b5_shoulderpress':
      b5 += 1
    if x == 'g1_shoulderpress':
      g1 += 1

  workout = {'B1 Shoulder Press': b1, 'B2 Shoulder Press': b2, 'B3 Shoulder Press': b3, 'B4 Shoulder Press': b4, 'B5 Shoulder Press': b5, 'G1 Shoulder Press': g1}
  #pos_result = max(workout, key=workout.get)
  #print("KNN Classifier: ", max(workout, key=workout.get))
  if max(workout, key=workout.get) == 'B1 Shoulder Press':
    print("Detected Posture Characteristics: ")
    print("Legs are detected to bend forward before lifting up the weight")
    print("Improper Posture Category: Jerking the Weight Up")
    print("Confident Level: ", str((b1/(b1+b2+b3+b4+b5+g1))*100), "%")
    print()
    print("Suggested Posture Collection: ")
    print("You are using the legs to drive the weight up, it is not shoulder press it is push press.")
    print("You need to keep your legs still while performing the exercise.")
    #pos_con = str((b1/(b1+b2+b3+b4+b5+g1))*100)
  if max(workout, key=workout.get) == 'B2 Shoulder Press':
    print("Detected Posture Characteristics: ")
    print("Torso bend over backwards and press the weight up.")
    print("Improper Posture Category: Not Keeping Torso Straight")
    print("Confident Level: ", str((b2/(b1+b2+b3+b4+b5+g1))*100), "%")
    print()
    print("Suggested Posture Collection: ")
    print("This might due to over over weight lifted, you should consider dropping some weight.")
    print("You need to keep your torso straight while performing the exercise.")
    print("Failure to keep your torso sraight might increase the risk on injury in spinal cord.")
    #pos_con = str((b2/(b1+b2+b3+b4+b5+g1))*100)
  if max(workout, key=workout.get) == 'B3 Shoulder Press':
    print("Detected Posture Characteristics: ")
    print("Your hands are not fully extended, which means this is only considered as half repetition.")
    print("Improper Posture Category: Lower Partial Range of Motion")
    print("Confident Level: ", str((b3/(b1+b2+b3+b4+b5+b6+g1))*100), "%")
    print()
    print("Suggested Posture Collection: ")
    print("In order to finish a full repetition, you need to make sure you hit full range of motion.")
    print("Lift the weight all the to the top of your body by fully extending your hands.")
    #pos_con = str((b3/(b1+b2+b3+b4+b5+g1))*100)
  if max(workout, key=workout.get) == 'B4 Shoulder Press':
    print("Detected Posture Characteristics: ")
    print("Your hands are not lower down to shoulder level, which means this is only considered as half repetition.")
    print("Improper Posture Category: Upper Partial Range of Motion")
    print("Confident Level: ", str((b4/(b1+b2+b3+b4+b5+b6+g1))*100), "%")
    print()
    print("Suggested Posture Collection: ")
    print("In order to finish a full repetition, you need to make sure you hit full range of motion.")
    print("Lift the weight all the to the top of your body by fully extending your hands and lowering them to same level of shoulders.")
    #pos_con = str((b4/(b1+b2+b3+b4+b5+g1))*100)
  if max(workout, key=workout.get) == 'B5 Shoulder Press':
    print("Detected Posture Characteristics: ")
    print("Your torso is not alligned straight and your arms extended slightly forward.")
    print("Improper Posture Category: Moving the Weight In Front instead of Up")
    print("Confident Level: ", str((b5/(b1+b2+b3+b4+b5+b6+g1))*100), "%")
    print()
    print("Suggested Posture Collection: ")
    print("Keep your hands and torso straight and make sure they are lined up properly when performing exercise.")    
    #pos_con = str((b5/(b1+b2+b3+b4+b5+g1))*100)
  if max(workout, key=workout.get) == 'G1 Shoulder Press':
    print("Detected Posture Characteristics: ")
    print("The back is straight, the hands are lifting the weight up with the same lining with back, performing full range of motion.")
    print("Improper Posture Category: None!")
    print("Congratulations! You are performing the correct posture, keep it up!")
    print("Confident Level: ", str((g1/(b1+b2+b3+b4+b5+g1))*100), "%")
    #pos_con = str((g1/(b1+b2+b3+b4+b5+g1))*100)

  #return pos_result, pos_con

def predict(video):
  cap = cv2.VideoCapture(video)
  hasFrame, frame = cap.read()

  #rotation = get_rotation(video)

  net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

  length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  final_xpoints = np.ones([length, nPoints])
  final_ypoints = np.ones([length, nPoints])

  recognition_trainingdata = np.zeros(8)
  bc_trainingdata = np.zeros(4)
  fr_trainingdata = np.zeros(2)
  sp_trainingdata = np.zeros(4)

  bc_left = 0
  bc_right = 0
  fr_left = 0
  fr_right = 0
  sp_left = 0
  sp_right = 0

  video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)

  fps = 1
  while cv2.waitKey(1) < 0:
    t = time.time()
    hasFrame, frame = cap.read()
    frameCopy = np.copy(frame)
    if not hasFrame:
      cv2.waitKey()
      break

    #print(str(fps) + '/' + str(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps += 1

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    # calculate the center of the image
    center = (frameWidth / 2, frameHeight / 2)

    #if rotation != 0:
      #print(rotation)
      #if rotation == 90:
        #M = cv2.getRotationMatrix2D(center, 270, 0.7)
        #frame = cv2.warpAffine(frame, M, (frameWidth, frameHeight), flags=cv2.INTER_LINEAR)
      #elif rotation == 270:
        #M = cv2.getRotationMatrix2D(center, 90, 0.7)
        #frame = cv2.warpAffine(frame, M, (frameWidth, frameHeight), flags=cv2.INTER_LINEAR)
      #else:
        #M = cv2.getRotationMatrix2D(center, 180, 1.0)
        #frame = cv2.warpAffine(frame, M, (frameHeight, frameWidth), flags=cv2.INTER_LINEAR)
      

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]
    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
      # confidence map of corresponding body's part.
      probMap = output[0, i, :, :]

      # Find global maxima of the probMap.
      minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
      
      # Scale the point to fit on the original image
      x = (frameWidth * point[0]) / W
      y = (frameHeight * point[1]) / H

      if prob > threshold : 
        cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        # Add the point to the list if the probability is greater than the threshold
        points.append((int(x), int(y)))
      else :
        points.append(None)

    ######################## RECOGNITION ########################
    ######################## FRONT RAISE ########################
    if points[4] and points[2] and points[8]:
      frontraise1 = cal_angle1(points[4], points[2], points[8])
    else:
      frontraise1 = 0

    if points[7] and points[5] and points[11]:
      frontraise2 = cal_angle1(points[7], points[5], points[11])
    else:
      frontraise2 = 0

    ######################## BICEP CURL ########################
    if points[4] and points[3] and points[2]:
      bicepcurl1 = cal_angle1(points[4], points[3], points[2])
    else:
      bicepcurl1 = 0

    if points[7] and points[6] and points[5]:
      bicepcurl2 = cal_angle1(points[7], points[6], points[5])
    else:
      bicepcurl2 = 0

    if points[3] and points[2] and points[8]:
      bicepcurl3 = cal_angle1(points[3], points[2], points[8])
    else:
      bicepcurl3 = 0

    if points[6] and points[5] and points[11]:
      bicepcurl4 = cal_angle1(points[6], points[5], points[11])
    else:
      bicepcurl4 = 0

    ###################### SHOULDER PRESS ######################
    if points[2] and points[4]:
      shoulderpress1 = cal_shoulderpress(points[4], points[2])
    else:
      shoulderpress1 = 0

    if points[5] and points[7]:
      shoulderpress2 = cal_shoulderpress(points[7], points[5])
    else:
      shoulderpress2 = 0

    recognition_feature = np.array([frontraise1, frontraise2, bicepcurl1, bicepcurl2, bicepcurl3, bicepcurl4, shoulderpress1, shoulderpress2]) #combine all feature arrays into one array

    recognition_trainingdata = np.vstack((recognition_trainingdata, recognition_feature))
    ######################## RECOGNITION ########################

    ######################## EVALUATION ########################
    ######################## BICEP CURL ########################
    if points[4] and points[3] and points[8] and points[2]: #tmr change from 8 to 9
      bicepcurl1 = cal_angle2(points[4], points[3], points[8], points[2])
    else:
      bicepcurl1 = 0
      bc_right = bc_right +1

    if points[9] and points[8] and points[8] and points[2]: #tmr change from 8 to 9
      bicepcurl2 = cal_angle2(points[9], points[8], points[8], points[2])
    else:
      bicepcurl2 = 0
      bc_right = bc_right +1

    if points[4] and points[3] and points[3] and points[2]:
      bicepcurl3 = cal_angle2(points[4], points[3], points[3], points[2])
    else:
      bicepcurl3 = 0
      bc_right = bc_right +1

    if points[3] and points[2] and points[8] and points[2]: #tmr change from 8 to 9
      bicepcurl4 = cal_angle2(points[3], points[2], points[8], points[2])
    else:
      bicepcurl4 = 0
      bc_right = bc_right +1

    if points[7] and points[6] and points[11] and points[5]: #tmr change from 11 to 12
      bicepcurl5 = cal_angle2(points[7], points[6], points[11], points[5])
    else:
      bicepcurl5 = 0
      bc_left = bc_left +1

    if points[12] and points[11] and points[11] and points[5]: #tmr change from 11 to 12
      bicepcurl6 = cal_angle2(points[12], points[11], points[11], points[5])
    else:
      bicepcurl6 = 0
      bc_left = bc_left +1

    if points[7] and points[6] and points[6] and points[5]: #tmr change from 11 to 12
      bicepcurl7 = cal_angle2(points[7], points[6], points[6], points[5])
    else:
      bicepcurl7 = 0
      bc_left = bc_left +1


    if points[6] and points[5] and points[11] and points[5]: #tmr change from 11 to 12
      bicepcurl8 = cal_angle2(points[6], points[5], points[11], points[5])
    else:
      bicepcurl8 = 0
      bc_left = bc_left +1

    if bc_right >= bc_left: #check whether right side or left side facing camera
      bc_feature = np.array([bicepcurl5, bicepcurl6, bicepcurl7, bicepcurl8]) #combine all feature arrays into one array
    else:
      bc_feature = np.array([bicepcurl1, bicepcurl2, bicepcurl3, bicepcurl4]) #combine all feature arrays into one array
    
    bc_trainingdata = np.vstack((bc_trainingdata, bc_feature))

    ######################## FRONT RAISE ########################
    if points[9] and points[8] and points[8] and points[2]: #tmr change from 8 to 9
      frontraise1 = cal_angle2(points[9], points[8], points[8], points[2])
    else:
      frontraise1 = 0
      fr_right = fr_right +1

    if points[4] and points[2] and points[8] and points[2]: #tmr change from 8 to 9
      frontraise2 = cal_angle2(points[4], points[2], points[8], points[2])
    else:
      frontraise2 = 0
      fr_right = fr_right +1

    if points[12] and points[11] and points[11] and points[5]:
      frontraise3 = cal_angle2(points[12], points[11], points[11], points[5])
    else:
      frontraise3 = 0
      fr_left = fr_left +1

    if points[7] and points[5] and points[11] and points[5]: #tmr change from 8 to 9
      frontraise4 = cal_angle2(points[7], points[5], points[11], points[5])
    else:
      frontraise4 = 0
      fr_left = fr_left +1

    if fr_right >= fr_left: #check whether right side or left side facing camera
      fr_feature = np.array([frontraise3, frontraise4]) #combine all feature arrays into one array
    else:
      fr_feature = np.array([frontraise1, frontraise2]) #combine all feature arrays into one array
    

    #cv2_imshow(frame)
    #print(feature)
    fr_trainingdata = np.vstack((fr_trainingdata, fr_feature))

    ######################## SHOULDER PRESS ########################
    if points[8] and points[2] and points[9] and points[8]:
      shoulderpress1 = cal_angle2(points[8], points[2], points[9], points[8])
    else:
      shoulderpress1 = 0
      sp_right = sp_right +1

    if points[9] and points[8] and points[10] and points[9]:
      shoulderpress2 = cal_angle2(points[9], points[8], points[10], points[9])
    else:
      shoulderpress2 = 0
      sp_right = sp_right +1

    if points[2] and points[4] and points[3] and points[4] and points[3][1] - points[4][1 != 0]:
      shoulderpress3 = round((points[2][1] - points[4][1]) / ((points[3][1] - points[4][1]) * 2), 3)
    else:
      shoulderpress3 = 0
      sp_right = sp_right +1

    if points[4] and points[2] and points[8] and points[2]:
      shoulderpress4 = cal_angle2(points[4], points[2], points[8], points[2])
    else:
      shoulderpress4 = 0
      sp_right = sp_right +1

    if points[11] and points[5] and points[12] and points[11]:
      shoulderpress5 = cal_angle2(points[11], points[5], points[12], points[11])
    else:
      shoulderpress5 = 0
      sp_left = sp_left +1

    if points[12] and points[11] and points[13] and points[12]:
      shoulderpress6 = cal_angle2(points[12], points[11], points[13], points[12])
    else:
      shoulderpress6 = 0
      sp_left = sp_left +1

    if points[7] and points[6] and points[6] and points[5] and points[6][1] - points[7][1] != 0:
      shoulderpress7 = round((points[5][1] - points[7][1]) / ((points[6][1] - points[7][1]) * 2), 3)
    else:
      shoulderpress7 = 0
      sp_left = sp_left +1

    if points[7] and points[5] and points[11] and points[5]:
      shoulderpress8 = cal_angle2(points[7], points[5], points[11], points[5])
    else:
      shoulderpress8 = 0
      sp_left = sp_left +1

    if sp_right >= sp_left: #check whether right side or left side facing camera
      sp_feature = np.array([shoulderpress5, shoulderpress6, shoulderpress7, shoulderpress8]) #combine all feature arrays into one array
    else:
      sp_feature = np.array([shoulderpress1, shoulderpress2, shoulderpress3, shoulderpress4]) #combine all feature arrays into one array
    
    sp_trainingdata = np.vstack((sp_trainingdata, sp_feature))
    

  ######################## EVALUATION ########################
    
  bc_trainingdata = bc_trainingdata[:-5]
  bc_trainingdata = bc_trainingdata[~np.all(bc_trainingdata == 0, axis=1)] # remove zero noise

  bc_trainingdata = np.append(bc_trainingdata,np.zeros([len(bc_trainingdata),7]),1) #train higher level feature vector 
  bc_b1 = 0 #counter, momentum
  bc_b2 = 0 #boolean, lower partial range 
  bc_b3 = 0 #boolean, upper partial range
  bc_b4 = 0 #counter, moving elbow
  bc_b5_1 = 0 #counter, bad core
  bc_b5_2 = 0 #counter, bad core

  for index in range(len(bc_trainingdata)):
    if bc_trainingdata[index][3] > -100 and bc_trainingdata[index][3] < -20:
      bc_b4 += 1
      
    if bc_trainingdata[index][1] > -155 and bc_trainingdata[index][1] != 0:
      bc_b5_1 += 1

    if bc_trainingdata[index][0] > 0:
      bc_b5_2 += 1

    if bc_trainingdata[index][0] > 0:
      bc_b1 += 1

  if np.amin(bc_trainingdata[:,2], axis = 0) >= 0 and np.amin(bc_trainingdata[:,0], axis = 0) >= -140:
    bc_b2 = 1

  if np.amin(bc_trainingdata[:,0], axis = 0) <= -140 and np.amax(bc_trainingdata[:,0], axis = 0) >= -90:
    bc_b3 = 1

  if bc_b1/len(bc_trainingdata) >= 0.05:
    for index in range(len(bc_trainingdata)):
      bc_trainingdata[index][4] = 1

  if bc_b2 == 1:
    for index in range(len(bc_trainingdata)):
      bc_trainingdata[index][5] = 1

  if bc_b3 == 1:
    for index in range(len(bc_trainingdata)):
      bc_trainingdata[index][6] = 1

  if bc_b4/len(bc_trainingdata) >= 0.2:
    for index in range(len(bc_trainingdata)):
      bc_trainingdata[index][7] = 1

  if bc_b5_1/len(bc_trainingdata) >= 0.1:
    for index in range(len(bc_trainingdata)):
      bc_trainingdata[index][8] = 1

  if bc_b5_2/len(bc_trainingdata) >= 0.05:
    for index in range(len(bc_trainingdata)):
      bc_trainingdata[index][9] = 1

  if video_duration <= 1.5:
    for index in range(len(bc_trainingdata)):
      bc_trainingdata[index][10] = 1
    

  fr_trainingdata = fr_trainingdata[:-5]
  fr_trainingdata = fr_trainingdata[~np.all(fr_trainingdata == 0, axis=1)] # remove zero noise

  fr_trainingdata = np.append(fr_trainingdata,np.zeros([len(fr_trainingdata),5]),1) #train higher level feature vector
  fr_b1 = 0 #counter
  fr_b2 = 0 #counter
  fr_b3 = 0 #boolean
  fr_b4 = 0 #boolean
  for index in range(len(fr_trainingdata)):
    if fr_trainingdata[index][0] > 10 and fr_trainingdata[index][0] < 60:
      fr_b1 += 1

    if fr_trainingdata[index][0] > 60:
      fr_b2 += 1

  if np.amin(fr_trainingdata[:,1], axis = 0) < -100:
    fr_b3 = 1

  if np.amin(fr_trainingdata[:,1], axis = 0) > -90:
    fr_b4 = 1

  if fr_b1/len(fr_trainingdata) >= 0.3:
    for index in range(len(fr_trainingdata)):
      fr_trainingdata[index][2] = 1

  if fr_b2/len(fr_trainingdata) >= 0.3:
    for index in range(len(fr_trainingdata)):
      fr_trainingdata[index][3] = 1

  if fr_b3 == 1:
    for index in range(len(fr_trainingdata)):
      fr_trainingdata[index][4] = 1

  if fr_b4 == 1:
    for index in range(len(fr_trainingdata)):
      fr_trainingdata[index][5] = 1

  if video_duration <= 1.5:
    for index in range(len(fr_trainingdata)):
      fr_trainingdata[index][6] = 1

  
  sp_trainingdata = sp_trainingdata[:-5]
  sp_trainingdata = sp_trainingdata[~np.all(sp_trainingdata == 0, axis=1)] # remove zero noise

  sp_trainingdata = np.append(sp_trainingdata,np.zeros([len(sp_trainingdata),7]),1) #train higher level feature vector
  sp_b1_1 = 0 #counter, moving leg
  sp_b1_2 = 0 #counter, moving leg
  sp_b2 = 0 #counter, bad core
  sp_b3 = 0 #counter, low partial
  sp_b4 = 0 #counter, up partial
  sp_b5 = 0 #counter, moving weight in front

  for index in range(len(sp_trainingdata)):
    if sp_trainingdata[index][0] < 0:
      sp_b1_1 += 1

    if sp_trainingdata[index][1] > 40 and sp_trainingdata[index][1] < 160:
      sp_b1_2 += 1

    if sp_trainingdata[index][0] > 100 and sp_trainingdata[index][0] < 160:
      sp_b2 += 1

    if sp_trainingdata[index][2] < 2:
      sp_b3 += 1

    if sp_trainingdata[index][2] > 2:
      sp_b4 += 1

    if sp_trainingdata[index][3] < -80 and sp_trainingdata[index][3] > -150:
      sp_b5 += 1

  if sp_b1_1/len(sp_trainingdata) >= 0.25:
    for index in range(len(sp_trainingdata)):
      sp_trainingdata[index][4] = 1

  if sp_b1_2/len(sp_trainingdata) >= 0.5:
    for index in range(len(sp_trainingdata)):
      sp_trainingdata[index][5] = 1

  if sp_b2/len(sp_trainingdata) >= 0.25:
    for index in range(len(sp_trainingdata)):
      sp_trainingdata[index][6] = 1
  
  if sp_b3/len(sp_trainingdata) >= 0.6:
    for index in range(len(sp_trainingdata)):
      sp_trainingdata[index][7] = 1

  if sp_b4/len(sp_trainingdata) >= 0.6:
    for index in range(len(sp_trainingdata)):
      sp_trainingdata[index][8] = 1

  if sp_b5/len(sp_trainingdata) >= 0.5:
    for index in range(len(sp_trainingdata)):
      sp_trainingdata[index][9] = 1

  if video_duration <= 1.5:
    for index in range(len(sp_trainingdata)):
      sp_trainingdata[index][10] = 1

  recognition_trainingdata = recognition_trainingdata[1:]

  scaler = MinMaxScaler() #perform feature scaling
  scaler.fit(recognition_trainingdata)
  MinMaxScaler(copy=True, feature_range=(0, 1))
  scaler.transform(recognition_trainingdata)

  loadedknn = pickle.load(open('Model/recognition.sav', "rb"))
  value = loadedknn.predict(recognition_trainingdata)
  sp = 0
  bc = 0
  fr = 0
  for x in value:
    if x == 'shoulderpress':
      sp += 1
    if x == 'bicepcurl':
      bc += 1
    if x == 'frontraise':
      fr += 1

  workout = {'Shoulder Press': sp, 'Bicep Curl': bc, 'Front Raise': fr}
  recog_result = max(workout, key=workout.get)
  print("Workout Classifier: ", max(workout, key=workout.get))
  if max(workout, key=workout.get) == 'Shoulder Press':
    print("Confident Level: ", str((sp/(sp+bc+fr))*100), "%")
    #recog_con = str((sp/(sp+bc+fr))*100)
    #pos_result, pos_con = shoulderpress(sp_trainingdata)
    shoulderpress(sp_trainingdata)
  if max(workout, key=workout.get) == 'Bicep Curl':
    print("Confident Level: ", str((bc/(sp+bc+fr))*100), "%")
    #recog_con = str((bc/(sp+bc+fr))*100)
    #pos_result, pos_con = bicepcurl(bc_trainingdata)
    bicepcurl(bc_trainingdata)
  if max(workout, key=workout.get) == 'Front Raise':
    print("Confident Level: ", str((fr/(sp+bc+fr))*100), "%")
    #recog_con = str((fr/(sp+bc+fr))*100)
    #pos_result, pos_con = frontraise(fr_trainingdata)
    frontraise(fr_trainingdata)

  #return recog_result, recog_con, pos_result, pos_con


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='predict arguments')
    parser.add_argument('video_path', type = str, help = 'Video Required')
    args = parser.parse_args()
    video_path = args.video_path

    predict(video_path)