import PIL
from PIL import ImageTk
import cv2
from tkinter import *
import numpy as np
import mediapipe
import tkinter.ttk as ttk

def angle_calculation (a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0] - b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180:
        angle = 360-angle

    return angle

DetectionNum = 2

md_drawing = mediapipe.solutions.drawing_utils
md_drawing_styles = mediapipe.solutions.drawing_styles
mp_pose = mediapipe.solutions.pose
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

root = Tk()
#root.minsize(width=1280, height=720)
#root.maxsize(width=1280, height=720)
#root.resizable(0,0)

circlesize = 7
VideoWidth = 1280
VideoHeight = 720
HipEgzersize = False
HandEgzersize = True

def OpenFootDetection():
    global DetectionNum
    DetectionNum = 0
def OpenIMUDetection():
    global DetectionNum
    DetectionNum = 1

TurnAround = False

c = Canvas(root, width=1280, height=720, bg = 'blue')
c.grid(row= 0, column=0)

ButtonFrame = LabelFrame(root, bg="red")
ButtonFrame.grid(row=0, column=1)
FootDetectButton = ttk.Button(ButtonFrame, text="Bel Egzersizi", command= OpenFootDetection, width=30)
FootDetectButton.grid(row=0, column=0)
IMUDetectionButton = ttk.Button(ButtonFrame, text="Kol Egzersizi", command= OpenIMUDetection, width=30)
IMUDetectionButton.grid(row=1, column=0)
IMUDetectionButton = ttk.Button(ButtonFrame, text="Kol Egzersizi", command= OpenIMUDetection, width=30)
IMUDetectionButton.grid(row=2, column=0)

root.bind('<Escape>', lambda e: root.quit())
blank = np.full((720, 1152 ,3), 0, np.uint8)
img2 = PIL.Image.fromarray(blank).resize((1152,720))
imgtk2 = ImageTk.PhotoImage(image=img2)  
cam = c.create_image(0,0, image=imgtk2, anchor=NW)

def show_frame():
    global c, imgtk
    ret , frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(
                min_detection_confidence = 0.5, 
                min_tracking_confidence = 0.5,
                model_complexity = 0
    ) as pose: 
        results = pose.process(frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    landmarks = results.pose_landmarks.landmark
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

    right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
    left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]

    RIGHT_KNEE_angle = int(angle_calculation(right_hip, right_knee, right_ankle))
    RIGHT_ANKLE_angle = int(angle_calculation(right_knee, right_ankle, right_foot_index))
    LEFT_KNEE_angle = int(angle_calculation(left_hip, left_knee, left_ankle))
    LEFT_ANKLE_angle = int(angle_calculation(left_knee, left_ankle, left_foot_index))

    if TurnAround == False:
        RIGHT_ELBOW_angle = int(angle_calculation(right_shoulder, right_elbow, right_wrist))
        RIGHT_SHOULDER_angle = int(angle_calculation(right_hip, right_shoulder, right_elbow))
        LEFT_ELBOW_angle = int(angle_calculation(left_shoulder, left_elbow, left_wrist))
        LEFT_SHOULDER_angle = int(angle_calculation(left_hip, left_shoulder, left_elbow))
        RIGHT_HIP_angle = int(angle_calculation(right_shoulder, right_hip, right_knee))
        LEFT_HIP_angle = int(angle_calculation(left_shoulder, left_hip, left_knee))

    #data.write(str(RIGHT_ELBOW_angle) + "," + str(RIGHT_SHOULDER_angle) + "," + str(RIGHT_HIP_angle) + "," + str(RIGHT_KNEE_angle) + "," + str(LEFT_KNEE_angle) + "\n")

    cv2.putText(frame, str(RIGHT_KNEE_angle), tuple(np.multiply(right_knee, [1300,760]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2, cv2.LINE_AA)
    cv2.putText(frame, str(RIGHT_ANKLE_angle), tuple(np.multiply(right_ankle, [1300,760]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2, cv2.LINE_AA)
    cv2.putText(frame, str(LEFT_KNEE_angle), tuple(np.multiply(left_knee, [1300,760]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2, cv2.LINE_AA)
    cv2.putText(frame, str(LEFT_ANKLE_angle), tuple(np.multiply(left_ankle, [1300,760]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2, cv2.LINE_AA)
    
    LEFT_HIP_POSITION_x = int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x *VideoWidth)
    LEFT_HIP_POSITION_y = int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y *VideoHeight)
    RIGHT_HIP_POSITION_x = int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x *VideoWidth)
    RIGHT_HIP_POSITION_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y *VideoHeight)

    LEFT_KNEE_POSITION_x = int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x *VideoWidth)
    LEFT_KNEE_POSITION_y = int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y *VideoHeight)
    RIGHT_KNEE_POSITION_x = int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x *VideoWidth)
    RIGHT_KNEE_POSITION_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y *VideoHeight)

    LEFT_ANKLE_POSITION_x = int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * VideoWidth)
    LEFT_ANKLE_POSITION_y = int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * VideoHeight)
    RIGHT_ANKLE_POSITION_x = int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * VideoWidth)
    RIGHT_ANKLE_POSITION_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * VideoHeight)

    LEFT_FOOT_INDEX_POSITION_x = int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x *VideoWidth)
    LEFT_FOOT_INDEX_POSITION_y = int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y *VideoHeight)
    RIGHT_FOOT_INDEX_POSITION_x = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x *VideoWidth)
    RIGHT_FOOT_INDEX_POSITION_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y *VideoHeight)
                            
    cv2.line(frame, (LEFT_HIP_POSITION_x, LEFT_HIP_POSITION_y), (LEFT_KNEE_POSITION_x, LEFT_KNEE_POSITION_y), (255, 255, 255), 2)
    cv2.line(frame, (LEFT_KNEE_POSITION_x, LEFT_KNEE_POSITION_y), (LEFT_ANKLE_POSITION_x, LEFT_ANKLE_POSITION_y), (255, 255, 255), 2)
    cv2.line(frame, (LEFT_ANKLE_POSITION_x, LEFT_ANKLE_POSITION_y), (LEFT_FOOT_INDEX_POSITION_x, LEFT_FOOT_INDEX_POSITION_y), (255, 255, 255), 2)

    if DetectionNum == 0:
        if RIGHT_KNEE_POSITION_y > LEFT_HIP_POSITION_y:
            UpperThanHip = False
        else:
            UpperThanHip = True
        if UpperThanHip == False:
            cv2.line(frame, (0, LEFT_HIP_POSITION_y), (1280, LEFT_HIP_POSITION_y), (0, 0, 255), 2)
        else:
            cv2.line(frame, (0, LEFT_HIP_POSITION_y), (1280, LEFT_HIP_POSITION_y), (0, 255, 0), 2)

    cv2.line(frame, (LEFT_HIP_POSITION_x, LEFT_HIP_POSITION_y), (RIGHT_HIP_POSITION_x, RIGHT_HIP_POSITION_y), (255, 255, 255), 2)

    cv2.line(frame, (RIGHT_HIP_POSITION_x, RIGHT_HIP_POSITION_y), (RIGHT_KNEE_POSITION_x, RIGHT_KNEE_POSITION_y), (255, 255, 255), 2)
    cv2.line(frame, (RIGHT_KNEE_POSITION_x, RIGHT_KNEE_POSITION_y), (RIGHT_ANKLE_POSITION_x, RIGHT_ANKLE_POSITION_y), (255, 255, 255), 2)
    cv2.line(frame, (RIGHT_ANKLE_POSITION_x, RIGHT_ANKLE_POSITION_y), (RIGHT_FOOT_INDEX_POSITION_x, RIGHT_FOOT_INDEX_POSITION_y), (255, 255, 255), 2)

    cv2.circle(frame, (LEFT_HIP_POSITION_x, LEFT_HIP_POSITION_y), circlesize, (0, 0, 255), cv2.FILLED)
    cv2.circle(frame, (LEFT_KNEE_POSITION_x, LEFT_KNEE_POSITION_y), circlesize, (0, 0, 255), cv2.FILLED)
    cv2.circle(frame, (LEFT_ANKLE_POSITION_x, LEFT_ANKLE_POSITION_y), circlesize, (0, 0, 255), cv2.FILLED)
    cv2.circle(frame, (LEFT_FOOT_INDEX_POSITION_x, LEFT_FOOT_INDEX_POSITION_y), circlesize, (0, 0, 255), cv2.FILLED)
    
    cv2.circle(frame, (RIGHT_HIP_POSITION_x, RIGHT_HIP_POSITION_y), circlesize, (0, 0, 255), cv2.FILLED)
    cv2.circle(frame, (RIGHT_KNEE_POSITION_x, RIGHT_KNEE_POSITION_y), circlesize, (0, 0, 255), cv2.FILLED)
    cv2.circle(frame, (RIGHT_ANKLE_POSITION_x, RIGHT_ANKLE_POSITION_y), circlesize, (0, 0, 255), cv2.FILLED)
    cv2.circle(frame, (RIGHT_FOOT_INDEX_POSITION_x, RIGHT_FOOT_INDEX_POSITION_y), circlesize, (0, 0, 255), cv2.FILLED)
    
    if TurnAround == False:
        cv2.putText(frame, str(LEFT_HIP_angle), tuple(np.multiply(left_hip, [1300,760]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2, cv2.LINE_AA)
        cv2.putText(frame, str(LEFT_ELBOW_angle), tuple(np.multiply(left_elbow, [1300,760]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2, cv2.LINE_AA)
        cv2.putText(frame, str(RIGHT_HIP_angle), tuple(np.multiply(right_hip, [1300,760]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2, cv2.LINE_AA)
        cv2.putText(frame, str(RIGHT_ELBOW_angle), tuple(np.multiply(right_elbow, [1300,760]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2, cv2.LINE_AA)
        cv2.putText(frame, str(RIGHT_SHOULDER_angle), tuple(np.multiply(right_shoulder, [1300,760]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2, cv2.LINE_AA)
        cv2.putText(frame, str(LEFT_SHOULDER_angle), tuple(np.multiply(left_shoulder, [1300,760]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2, cv2.LINE_AA)
        #cv2.putText(frame, "Lutfen kollarinizi acip donun! 卐", (360, 30) , cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,255), 1, cv2.LINE_AA)
        
        LEFT_SHOULDER_POSITION_x = int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * VideoWidth)
        LEFT_SHOULDER_POSITION_y = int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * VideoHeight)
        RIGHT_SHOULDER_POSITION_x = int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * VideoWidth)
        RIGHT_SHOULDER_POSITION_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * VideoHeight)

        LEFT_ELBOW_POSITION_x = int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * VideoWidth)
        LEFT_ELBOW_POSITION_y = int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * VideoHeight)
        RIGHT_ELBOW_POSITION_x = int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * VideoWidth)
        RIGHT_ELBOW_POSITION_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * VideoHeight)

        LEFT_WRIST_POSITION_x = int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * VideoWidth)
        LEFT_WRIST_POSITION_y = int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * VideoHeight)
        RIGHT_WRIST_POSITION_x = int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * VideoWidth)
        RIGHT_WRIST_POSITION_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * VideoHeight)
        
        if DetectionNum == 1:
            if LEFT_WRIST_POSITION_x < (LEFT_SHOULDER_POSITION_x + ((RIGHT_SHOULDER_POSITION_x  - LEFT_SHOULDER_POSITION_x)/2)) or RIGHT_WRIST_POSITION_x > (LEFT_SHOULDER_POSITION_x + ((RIGHT_SHOULDER_POSITION_x  - LEFT_SHOULDER_POSITION_x)/2)):
                LefterRighterHand = True
            else:
                LefterRighterHand = False
            if LefterRighterHand == False:
                cv2.line(frame, (LEFT_SHOULDER_POSITION_x + (int)((RIGHT_SHOULDER_POSITION_x  - LEFT_SHOULDER_POSITION_x)/2), 0), (LEFT_SHOULDER_POSITION_x + (int)((RIGHT_SHOULDER_POSITION_x - LEFT_SHOULDER_POSITION_x)/2), VideoHeight), (0, 0, 255), 2)
            else:
                cv2.line(frame, (LEFT_SHOULDER_POSITION_x + (int)((RIGHT_SHOULDER_POSITION_x - LEFT_SHOULDER_POSITION_x)/2), 0), (LEFT_SHOULDER_POSITION_x + (int)((RIGHT_SHOULDER_POSITION_x - LEFT_SHOULDER_POSITION_x)/2), VideoHeight), (0, 255, ), 2)
            
        cv2.line(frame, (LEFT_SHOULDER_POSITION_x, LEFT_SHOULDER_POSITION_y), (LEFT_HIP_POSITION_x, LEFT_HIP_POSITION_y), (255, 255, 255), 2)
        cv2.line(frame, (LEFT_SHOULDER_POSITION_x, LEFT_SHOULDER_POSITION_y), (RIGHT_SHOULDER_POSITION_x, RIGHT_SHOULDER_POSITION_y), (255, 255, 255), 2)
        cv2.line(frame, (LEFT_SHOULDER_POSITION_x, LEFT_SHOULDER_POSITION_y), (LEFT_ELBOW_POSITION_x, LEFT_ELBOW_POSITION_y), (255, 255, 255), 2)
        cv2.line(frame, (LEFT_ELBOW_POSITION_x, LEFT_ELBOW_POSITION_y), (LEFT_WRIST_POSITION_x, LEFT_WRIST_POSITION_y), (255, 255, 255), 2)

        cv2.line(frame, (RIGHT_SHOULDER_POSITION_x, RIGHT_SHOULDER_POSITION_y), (RIGHT_HIP_POSITION_x, RIGHT_HIP_POSITION_y), (255, 255, 255), 2)
        cv2.line(frame, (RIGHT_SHOULDER_POSITION_x, RIGHT_SHOULDER_POSITION_y), (RIGHT_ELBOW_POSITION_x, RIGHT_ELBOW_POSITION_y), (255, 255, 255), 2)
        cv2.line(frame, (RIGHT_ELBOW_POSITION_x, RIGHT_ELBOW_POSITION_y), (RIGHT_WRIST_POSITION_x, RIGHT_WRIST_POSITION_y), (255, 255, 255), 2)

        cv2.circle(frame, (LEFT_SHOULDER_POSITION_x, LEFT_SHOULDER_POSITION_y), circlesize, (0, 0, 255), cv2.FILLED)
        cv2.circle(frame, (LEFT_ELBOW_POSITION_x, LEFT_ELBOW_POSITION_y), circlesize, (0, 0, 255), cv2.FILLED)
        cv2.circle(frame, (LEFT_WRIST_POSITION_x, LEFT_WRIST_POSITION_y), circlesize, (0, 0, 255), cv2.FILLED)

        cv2.circle(frame, (RIGHT_SHOULDER_POSITION_x, RIGHT_SHOULDER_POSITION_y), circlesize, (0, 0, 255), cv2.FILLED)
        cv2.circle(frame, (RIGHT_ELBOW_POSITION_x, RIGHT_ELBOW_POSITION_y), circlesize, (0, 0, 255), cv2.FILLED)
        cv2.circle(frame, (RIGHT_WRIST_POSITION_x, RIGHT_WRIST_POSITION_y), circlesize, (0, 0, 255), cv2.FILLED)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = PIL.Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    c.itemconfig(cam, image=imgtk) #must use same ImageTk object
    c.grid(row=0, column=0)
    c.after(10, show_frame)

show_frame()
root.mainloop()