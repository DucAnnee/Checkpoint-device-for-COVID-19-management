# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from threading import Thread
import serial
from time import strftime
from datetime import datetime

import os

from pygame import mixer
import time
from tensorflow.python.keras.layers.preprocessing.category_encoding import COUNT

from tensorflow.python.ops.gen_nn_ops import fractional_avg_pool
mixer.init()

# Grab path to current working directory
CWD_PATH = os.getcwd()
audio_path = os.path.join(CWD_PATH,"Audio_QLDT")
out_img_path = os.path.join(CWD_PATH,"Out_Img")
csv_path = os.path.join(CWD_PATH,"System_1.xlsx")
sys_img_path = os.path.join(CWD_PATH,"System_images")

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='He Thong Quan Ly Dich te')
parser.add_argument('--port_ard', required=False,
                    metavar="",default="/dev/ttyUSB0",
                    help='Port number of Arduino (connect with Arduino 1)')
parser.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
parser.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
parser.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(parser.parse_args())

## Viet cac function can thiet
###################################################
##                  AUDIO                        ##
###################################################
## Phat loa
def phat_loa(audio_file_name):
    audio_file_name = os.path.join(audio_path,audio_file_name)
    if mixer.music.get_busy():
        cho_loa = 1
        while cho_loa ==1:
            if not mixer.music.get_busy():
                mixer.music.load(audio_file_name)
                mixer.music.play()
                cho_loa = 0
    elif not mixer.music.get_busy():
        mixer.music.load(audio_file_name)
        mixer.music.play()

## Phat loa va cho den ket thuc
def phat_loa_untill_end(audio_file_name):
    audio_file_name = os.path.join(audio_path,audio_file_name)
    if mixer.music.get_busy():
        cho_loa = 1
        while cho_loa ==1:
            if not mixer.music.get_busy():
                mixer.music.load(audio_file_name)
                mixer.music.play()
                cho_loa = 0
                phat = 1 
                while phat == 1:
                    if not mixer.music.get_busy():
                        phat=0            
                        time.sleep(0.05)
    elif not mixer.music.get_busy():
        mixer.music.load(audio_file_name)
        mixer.music.play()
        phat = 1
        while phat==1:
            if not mixer.music.get_busy():
                phat=0            
            time.sleep(0.05)

def phat_loa_cam_shutter_sound():
    phat_loa("cam_shutter_sound.wav")
    # phat_loa_untill_end("cam_shutter_sound.wav")
    time.sleep(0.5)

###################################################
##                  VISUAL                       ##
###################################################
## Show the image in fixed wondow   
system_name_location = [640,0]
info_show_location = [640,480]
info_img_width = 320
def showInFixedWindow(winname, img, x, y):
    cv2.namedWindow(winname, flags=cv2.WINDOW_GUI_NORMAL+ cv2.WINDOW_AUTOSIZE)        # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    cv2.imshow(winname,img)

def visualize(face_frame):
    showInFixedWindow("Face_Camera", face_frame,face_show_location[0],face_show_location[1])

def visualize_ten_hethong():
    img = cv2.imread(os.path.join(sys_img_path,"ten_hethong.png"))
    img = imutils.resize(img, width = info_img_width)
    showInFixedWindow('System Name',img, system_name_location[0], system_name_location[1])
    cv2.waitKey(1)

def visualize_cam_on():
    img = cv2.imread(os.path.join(sys_img_path,"cam_on.png"))
    img = imutils.resize(img, width = info_img_width)
    showInFixedWindow('System Notice',img, info_show_location[0], info_show_location[1])
    cv2.waitKey(1)

###################################################
##              VISUALIZE & AUDIO                ##
###################################################
# 1. Dung vao vi tri 2 ban chan
def visualize_1_dung_2banchan():
    cv2.destroyWindow("System Notice")
    img = cv2.imread(os.path.join(sys_img_path,"1_dung_2banchan.png"))
    img = imutils.resize(img, width = info_img_width)
    showInFixedWindow('System Notice 1',img, info_show_location[0], info_show_location[1])
    cv2.waitKey(1)

def phat_loa_1_dung_2banchan():
    visualize_1_dung_2banchan()
    phat_loa("1_dung_2banchan.wav")
    # phat_loa_untill_end("1_dung_2banchan.wav")
    print("1. Da phat loa chao mung va dung vao vi tri 2 ban chan")
    time.sleep(0.5)
#-------------------------------------------------#
# 2. Deo khau trang
def visualize_2_deo_khautrang():
    cv2.destroyWindow("System Notice 1")
    img = cv2.imread(os.path.join(sys_img_path,"2_deo_khautrang.png"))
    img = imutils.resize(img, width = info_img_width)
    showInFixedWindow('System Notice 2',img, info_show_location[0], info_show_location[1])
    cv2.waitKey(1)
def phat_loa_2_deo_khautrang():
    visualize_2_deo_khautrang()
    phat_loa("2_deo_khautrang.wav")
    # phat_loa_untill_end("2_deo_khautrang.wav")
    print("2. Da phat loa nhac nho deo khau trang")
    time.sleep(0.5)
#-------------------------------------------------#
# 2.0 Nhin thang vao camera
def visualize_20_nhinthang_camera():
    cv2.destroyWindow("System Notice 1")
    cv2.destroyWindow("System Notice 2")
    img = cv2.imread(os.path.join(sys_img_path,"20_nhinthang_camera.png"))
    img = imutils.resize(img, width = info_img_width)
    showInFixedWindow('System Notice 20',img, info_show_location[0], info_show_location[1])
    cv2.waitKey(1)
def phat_loa_20_nhinthang_camera():
    visualize_20_nhinthang_camera()
    phat_loa("20_nhinthang_camera.wav")
    # phat_loa_untill_end("20_nhinthang_camera.wav")
    print("2.0 Da phat loa nhin thang vao camera")
    time.sleep(0.5)
#-------------------------------------------------#
# 3. Dua ma QR
def visualize_3_dua_maQR():
    cv2.destroyWindow("System Notice 20")
    img = cv2.imread(os.path.join(sys_img_path,"3_dua_maQR.png"))
    img = imutils.resize(img, width = info_img_width)
    showInFixedWindow('System Notice 3',img, info_show_location[0], info_show_location[1])
    cv2.waitKey(1)
def phat_loa_3_dua_maQR():
    visualize_3_dua_maQR()
    phat_loa("3_dua_maQR.wav")
    # phat_loa_untill_end("3_dua_maQR.wav")
    print("3. Da phat loa dua ma QR den truoc he thong")
    time.sleep(0.5)
#-------------------------------------------------#
# 4. Dua dien thoai lai gan
def visualize_4_dua_dt_gan():
    cv2.destroyWindow("System Notice 3")
    img = cv2.imread(os.path.join(sys_img_path,"4_dua_dt_gan.png"))
    img = imutils.resize(img, width = info_img_width)
    showInFixedWindow('System Notice 4',img, info_show_location[0], info_show_location[1])
    cv2.waitKey(1)
def phat_loa_4_dua_dt_gan():
    visualize_4_dua_dt_gan()
    phat_loa("4_dua_dt_gan.wav")
    # phat_loa_untill_end("4_dua_dt_gan.wav")
    print("4. Da phat loa dua dien thoai lai gan")
    time.sleep(0.5)
#-------------------------------------------------#
# 41. Tao lai ma QR
def visualize_41_taolai_maQR():
    cv2.destroyWindow("System Notice 3")
    cv2.destroyWindow("System Notice 4")
    img = cv2.imread(os.path.join(sys_img_path,"41_taolai_maQR.png"))
    img = imutils.resize(img, width = info_img_width)
    showInFixedWindow('System Notice 4.1',img, info_show_location[0], info_show_location[1])
    cv2.waitKey(1)
def phat_loa_41_taolai_maQR():
    visualize_41_taolai_maQR()
    phat_loa("41_taolai_maQR.wav")
    # phat_loa_untill_end("41_taolai_maQR.wav")
    print("4.1 Da phat loa tao lai ma QR")
    time.sleep(0.5)
#-------------------------------------------------#
# 42. Di qua vung do
def visualize_42_diqua_vungdo():
    cv2.destroyWindow("System Notice 3")
    cv2.destroyWindow("System Notice 4")
    cv2.destroyWindow("System Notice 4.1")
    img = cv2.imread(os.path.join(sys_img_path,"42_diqua_vungdo.png"))
    img = imutils.resize(img, width = info_img_width)
    showInFixedWindow('System Notice 4.2',img, info_show_location[0], info_show_location[1])
    cv2.waitKey(1)
def phat_loa_42_diqua_vungdo():
    visualize_42_diqua_vungdo()
    phat_loa("42_diqua_vungdo.wav")
    # phat_loa_untill_end("42_diqua_vungdo.wav")
    print("4.2 Da phat loa canh bao di qua vung do")
    time.sleep(0.5)
#-------------------------------------------------#
# 5. Do nhiet do
def visualize_5_do_nhietdo():
    cv2.destroyWindow("System Notice 3")
    cv2.destroyWindow("System Notice 4")
    cv2.destroyWindow("System Notice 4.1")
    cv2.destroyWindow("System Notice 4.2")
    img = cv2.imread(os.path.join(sys_img_path,"5_do_nhietdo.png"))
    img = imutils.resize(img, width = info_img_width)
    showInFixedWindow('System Notice 5',img, info_show_location[0], info_show_location[1])
    cv2.waitKey(1)
def phat_loa_5_do_nhietdo():
    visualize_5_do_nhietdo()
    phat_loa("5_do_nhietdo.wav")
    # phat_loa_untill_end("5_do_nhietdo.wav")
    print("5. Da phat loa yeu cau do nhiet do")
    time.sleep(0.5)
#-------------------------------------------------#
# 6. Nhiet do binh thuong
def visualize_6_nhietdo_binhthuong():
    cv2.destroyWindow("System Notice 5")
    img = cv2.imread(os.path.join(sys_img_path,"6_nhietdo_binhthuong.png"))
    img = imutils.resize(img, width = info_img_width)
    showInFixedWindow('System Notice 6',img, info_show_location[0], info_show_location[1])
    cv2.waitKey(1)
def phat_loa_6_nhietdo_binhthuong():
    visualize_6_nhietdo_binhthuong()
    phat_loa("6_nhietdo_binhthuong.wav")
    # phat_loa_untill_end("6_nhietdo_binhthuong.wav")
    print("6. Da phat loa nhiet do binh thuong")
    time.sleep(0.5)
#-------------------------------------------------#
# 7. Dau hieu sot
def visualize_7_dauhieu_sot():
    cv2.destroyWindow("System Notice 5")
    img = cv2.imread(os.path.join(sys_img_path,"7_dauhieu_sot.png"))
    img = imutils.resize(img, width = info_img_width)
    showInFixedWindow('System Notice 7',img, info_show_location[0], info_show_location[1])
    cv2.waitKey(1)
def phat_loa_7_dauhieu_sot():
    visualize_7_dauhieu_sot()
    phat_loa("7_dauhieu_sot.wav")
    # phat_loa_untill_end("7_dauhieu_sot.wav")
    print("6. Da phat loa co dau hieu sot")
    time.sleep(0.5)

###################################################
##                ULTIS FUNCTION                 ##
###################################################
## Load the face detector
print("[INFO] loading facial detector...")
detector = dlib.get_frontal_face_detector()  
COUNTER_FACE_THRES = 30
COUNTER_FACE_QLDT_THRES = 70
COUNT_KHONG_KHAU_TRANG_THRES = 150
COUNT_QR_THRES = 50
src_cam = 0
face_show_location = [0,0]
###################################################
##                FACENET                 ##
###################################################
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

#Nhan dang deo khau trang
def nhan_mat_chup_hinh_for_mask(faceNet, COUNTER_FACE_THRES,out_img_name): #qldt_row,
    # grab the dimensions of the frame and then construct a blob
	# from it
    visualize_20_nhinthang_camera()
    COUNTER_FACE = 0
    while COUNTER_FACE < COUNTER_FACE_THRES:
        frame_org = face_cam.read()
        frame = imutils.resize(frame_org, width=640)    #450, cang nho xu ly cang nhanh
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
            (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        faceNet.setInput(blob)
        detections = faceNet.forward()
        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > args["confidence"]:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))
        
        for box in locs:
            (x, y, w, h) = box
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 1)

        timestamp = datetime.now()
        ts = timestamp.strftime("%A %d %B %Y %H:%M:%S")
        cv2.putText(frame, ts, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, "{:.0f}".format(COUNTER_FACE), (590, 300),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.putText(frame, "VUI LONG NHIN THANG VAO CAMERA", (10, 460),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        visualize(frame)
        key = cv2.waitKey(1) & 0xFF
		# if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
            # do a bit of cleanup

        if len(locs) > 0:
            COUNTER_FACE += 1
        else:
            COUNTER_FACE=0
            # phat_loa("nhin_camera.wav")
        print(COUNTER_FACE)

    else:
        print("Da chup hinh xong")
        # cv2.destroyWindow("Face_Camera")
        name = os.path.join(out_img_path, str(out_img_name) + '.jpg')
        print(name)
        # qldt_row.append(name)
        cv2.imwrite(name, frame_org)  
        phat_loa_cam_shutter_sound()
        print("Save image successfully!")
        return name

def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

def nhan_dang_khau_trang():
    frame_org = face_cam.read()
    frame = imutils.resize(frame_org, width=640)
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
    for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        sta_khau_trang = 1 if mask > withoutMask else 0
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        print(label)

		# display the label and bounding box rectangle on the output
		# frame
        cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        timestamp = datetime.now()
        ts = timestamp.strftime("%A %d %B %Y %H:%M:%S")
        cv2.putText(frame, ts, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

	# show the output frame
    visualize(frame)
    return sta_khau_trang

## Nhan dang mat chup hinh
def Nhan_mat_chup_hinh(COUNTER_FACE_THRES,qldt_row,out_img_name):
    visualize_20_nhinthang_camera()
    COUNTER_FACE = 0
    while COUNTER_FACE < COUNTER_FACE_THRES:
        frame_org = face_cam.read()
        frame = imutils.resize(frame_org, width=480)    #450, cang nho xu ly cang nhanh
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4,4))
        gray = clahe.apply(gray)
        # Nhan dien va trich xuat khuon mat
        rects = detector(gray, 0)
        for rect in rects:
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        
        timestamp = datetime.now()
        ts = timestamp.strftime("%A %d %B %Y %H:%M:%S")
        cv2.putText(frame, ts, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        cv2.putText(frame, "{:.0f}".format(COUNTER_FACE), (590, 40),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        # cv2.putText(frame, "VUI LONG NHIN THANG VAO CAMERA", (10, 460),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        visualize(frame)
        key = cv2.waitKey(1) & 0xFF
		# if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
            # do a bit of cleanup

        if len(rects) > 0:
            COUNTER_FACE += 1
        else:
            COUNTER_FACE=0
            # phat_loa("nhin_camera.wav")
        print(COUNTER_FACE)

    else:
        print("Da chup hinh xong")
        # cv2.destroyWindow("Face_Camera")
        name = os.path.join(out_img_path, str(out_img_name) + '.jpg')
        print(name)
        qldt_row.append(name)
        cv2.imwrite(name, frame_org)  
        phat_loa_cam_shutter_sound()
        print("Save image successfully!")
        return name

## Nhan dang mat bat dau quan ly dich te
def Nhan_mat_QLDT(COUNTER_FACE_QLDT_THRES):
    COUNTER_FACE = 0
    while COUNTER_FACE < COUNTER_FACE_QLDT_THRES:
        frame_org = face_cam.read()
        frame = imutils.resize(frame_org, width=480)    #450, cang nho xu ly cang nhanh
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4,4))
        gray = clahe.apply(gray)
        # Nhan dien va trich xuat khuon mat
        rects = detector(gray, 0)
        for rect in rects:
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

         #### Add text
        timestamp = datetime.now()
        ts = timestamp.strftime("%A %d %B %Y %H:%M:%S")
        # cv2.putText(frame, "HE THONG KHAI BAO Y TE TU DONG", (10, 40),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(frame, ts, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, "{:.0f}".format(COUNTER_FACE), (590, 40),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.putText(frame, "VUI LONG NHIN THANG VAO CAMERA", (10, 460),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        visualize(frame)

        key = cv2.waitKey(1) & 0xFF
		# if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
            # do a bit of cleanup
        if len(rects) > 0:
            COUNTER_FACE += 1
        else:
            COUNTER_FACE=0
            # phat_loa("nhin_camera.wav")
        print(COUNTER_FACE)
    else:
        # cv2.destroyWindow("Face_Camera")
        phat_loa_1_dung_2banchan()
        print("Phat hien nguoi den truoc he thong QLDT")
        bat_dau_QLDT = 1
        return bat_dau_QLDT

## Nhan dang mat chup hinh va bat dau quan ly dich te
def Nhanmat_chuphinh_QLDT(COUNTER_FACE_QLDT_THRES,qldt_row,out_img_name):
    COUNTER_FACE = 0
    while COUNTER_FACE < COUNTER_FACE_QLDT_THRES:
        frame_org = face_cam.read()
        frame = imutils.resize(frame_org, width=480)    #450, cang nho xu ly cang nhanh
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4,4))
        gray = clahe.apply(gray)
        # Nhan dien va trich xuat khuon mat
        rects = detector(gray, 0)
        for rect in rects:
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

         #### Add text
        timestamp = datetime.now()
        ts = timestamp.strftime("%A %d %B %Y %H:%M:%S")
        # cv2.putText(frame, "HE THONG KHAI BAO Y TE TU DONG", (10, 40),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(frame, ts, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, "{:.0f}".format(COUNTER_FACE), (590, 40),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.putText(frame, "VUI LONG NHIN THANG VAO CAMERA", (10, 460),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        visualize(frame)

        key = cv2.waitKey(1) & 0xFF
		# if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
            # do a bit of cleanup
        if len(rects) > 0:
            COUNTER_FACE += 1
        else:
            COUNTER_FACE=0
            # phat_loa("nhin_camera.wav")
        print(COUNTER_FACE)
    else:
        # cv2.destroyWindow("Face_Camera")
        phat_loa_1_dung_2banchan()
        print("Phat hien nguoi den truoc he thong QLDT")
        bat_dau_QLDT = 1

        # visualize_20_nhinthang_camera()
        print("Da chup hinh xong")
        # cv2.destroyWindow("Face_Camera")
        name = os.path.join(out_img_path, str(out_img_name) + '.jpg')
        print(name)
        qldt_row.append(name)
        cv2.imwrite(name, frame_org)  
        phat_loa_cam_shutter_sound()
        print("Save image successfully!")
        return bat_dau_QLDT, name

###################################################
##                QR CODE SCANNER                ##
###################################################    
from pyzbar import pyzbar    
def doc_ma_QR():
    text_split = []
    found = set()
	# grab the frame from the threaded video stream and resize it to
	# have a maximum width of 400 pixels
    frame_org = face_cam.read()
    frame = imutils.resize(frame_org, width=480)
	# find the barcodes in the frame and decode each of the barcodes
    barcodes = pyzbar.decode(frame)

	# loop over the detected barcodes
    for barcode in barcodes:
		# extract the bounding box location of the barcode and draw
		# the bounding box surrounding the barcode on the image
        (x, y, w, h) = barcode.rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

		# the barcode data is a bytes object so if we want to draw it
		# on our output image we need to convert it to a string first
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type

		# draw the barcode data and barcode type on the image
		# text = "{} ({})".format(barcodeData, barcodeType)
        text = barcodeData
        # print(text)
        text_split = text.split("_")
        print(text_split)
        print(len(text_split))
        cv2.putText(frame, barcodeType, (x, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

		# if the barcode text is currently not in our CSV file, write
		# the timestamp + barcode to disk and update the set
        if barcodeData not in found:
            found.add(barcodeData)

	# show the output frame
    visualize(frame)
    return text_split
   
###################################################
##             INTERFACE WITH ARDUINO            ##
###################################################
## Ket noi voi arduino 
try:                                                                                  
    arduino_module = serial.Serial(args.port_ard, 9600, timeout = 0.05)                         
    arduino_module.flush()
    print("Arduino connected successfully!")                                            
except:                                                                               
    print("Please check the port again") 

## Send message to Arduino
def send_message_to_arduino(arduino_module, message):
    #red_zone
    #start_measure
    #raspi_da_doc_binh_thuong
    #raspi_da_doc_canhbao
    print("Message: ", message)
    arduino_module.write(message.encode())
    print("-Sent data to Arduino-")
    time.sleep(0.05)

## Read message from Arduino
def read_message_from_arduino(arduino_module):
    #start_process
    # RPi_tempC = "raspi_binhthuong" + rounded_val_tempC
    # RPi_tempC = "raspi_quanhiet" + rounded_val_tempC
    # ard_finish
    message = arduino_module.readline() # doc tin hieu ve
    message = message.decode("utf-8").rstrip('\r\n') 
    print("Data from Arduino: ", message)
    return message

###################################################
##             INTERFACE WITH CAMERA             ##
###################################################
## 4. Ket noi camera
# start the video stream thread
# sudo rmmod uvcvideo
# sudo modprobe uvcvideo timeout=2000
# v4l2-ctl --list-devices
try:
    print("[INFO] starting camera ...")
    face_cam = VideoStream(src=src_cam).start()
    time.sleep(1.0)
    print("Camera connect Successfully!")
except:
    print("Connect not successfully!!!")
    pass

###################################################
##                  EXCEL OUTPUT                 ##
###################################################
# Luu file excel
import pandas as pd # ho viec doc thong tin tu file excel
header = ['Address_Code','Check-in Place','Name','CMND','Temperature','Collecting time','FaceImage_Path']

###################################################
##               CHUONG TRINH CHINH              ##
###################################################
## Load hinh thong tin he thong
visualize_ten_hethong()
# # loop over frames from the video streams
while True:
    qldt_row = ['A1','Bệnh viện Phường 1']
    global_ts = datetime.now().strftime("%H_%M_%d_%m_%Y")
    step = 1
    while step == 1:
        # Chờ đọc tín hiệu từ arduino
        #     ##Check Serial
        if arduino_module.in_waiting > 0:
            msg = read_message_from_arduino(arduino_module)
            if (msg == "start_process"):
                # chao_mung()
                print("Tin hieu la nhan dc la: ", msg)
                # Phát loa chào mừng và yêu cầu đứng vào vị trí 2 bàn chân
                phat_loa_1_dung_2banchan()
                # Chụp hình, lưu vào máy, trả về đường dẫn lưu hình
                face_path = nhan_mat_chup_hinh_for_mask(faceNet, COUNTER_FACE_THRES,global_ts)
                # Kiểm tra trạng thái đeo khẩu trang
                sta_khau_trang = nhan_dang_khau_trang()
                print("sta_khau_trang: ",sta_khau_trang)
                # Nếu phát hiện không đeo khẩu trang
                if sta_khau_trang ==0:
                    # Nhắc nhở đeo khẩu trang
                    phat_loa_2_deo_khautrang()
                    # Gán biến yêu cầu đeo khẩu trang lên 1 và bắt đầu đếm số frame có đeo khẩu trang hay không
                    yeu_cau_deo_khau_trang = 1
                    COUNT_KHONG_KHAU_TRANG = 0
                while yeu_cau_deo_khau_trang == 1:
                    sta_khau_trang = nhan_dang_khau_trang()
                    if sta_khau_trang == 0:
                        COUNT_KHONG_KHAU_TRANG +=1
                        print("COUNT_KHONG_KHAU_TRANG: ",COUNT_KHONG_KHAU_TRANG)
                    else:
                        yeu_cau_deo_khau_trang = 0
                        print("yeu_cau_deo_khau_trang: ",yeu_cau_deo_khau_trang)
                    # Nhắc nhở đeo khẩu trang lại khi chờ qua ngưỡng đặt trước
                    if COUNT_KHONG_KHAU_TRANG > COUNT_KHONG_KHAU_TRANG_THRES:
                        phat_loa_2_deo_khautrang()
                        COUNT_KHONG_KHAU_TRANG = 0
                    
                ## Yêu cầu đưa mã QR
                phat_loa_3_dua_maQR()
                cho_quet_maQR = 1
                COUNT_QR = 0
                while cho_quet_maQR==1:
                    text_split = doc_ma_QR()

                    if len(text_split)>0:
                        # Xét xem có đi qua vùng đỏ ko
                        sta_red_blue = text_split[3]
                        if sta_red_blue == "RED":
                            cho_quet_maQR = 0
                            phat_loa_42_diqua_vungdo()
                            send_message_to_arduino(arduino_module,"red_zone")
                        if sta_red_blue == "BLUE":
                            QR_time = text_split[2]
                            print(QR_time)
                            current_date = datetime.now().strftime("%d/%m/%Y")
                            print(current_date)
                            current_hour = datetime.now().strftime("%H")
                            print(current_hour)
                            current_min = datetime.now().strftime("%M")
                            print(current_min)
                            date_time_QR = text_split[2].split(" ")
                            print(date_time_QR)
                            time_hm_QR= date_time_QR[0]
                            print(time_hm_QR)
                            # Tách giờ và phút từ thời gian đọc lại từ mã
                            hour_QR=time_hm_QR.split(":")[0]
                            min_QR=time_hm_QR.split(":")[1]
                            date_QR = date_time_QR[1]
                            print(date_QR) 
                            # Xét thời gian có bị lố ko 
                            if current_date != date_QR or current_hour != hour_QR or (int(current_hour)-int(min_QR))>3:
                                phat_loa_41_taolai_maQR()
                            else:
                                cho_quet_maQR = 0
                                send_message_to_arduino(arduino_module,"start_measure")
                                step = 2
                                phat_loa_5_do_nhietdo()
                    else: 
                        COUNT_QR +=1
                    if COUNT_QR > COUNT_QR_THRES:
                        phat_loa_4_dua_dt_gan()
                        COUNT_QR = 0
    while step == 2: 
        if arduino_module.in_waiting > 0:
            msg = read_message_from_arduino(arduino_module)
            if (msg.startswith('raspi_binhthuong')):
                print("Tin hieu la nhan dc la: ", msg) 
                phat_loa_6_nhietdo_binhthuong()
                send_message_to_arduino(arduino_module,"raspi_da_doc_binh_thuong")
                temp_C = int(msg[-4:])/100
                step = 3
    
    while step == 3:
        if arduino_module.in_waiting > 0:
            msg = read_message_from_arduino(arduino_module)
            if (msg.startswith('raspi_quanhiet')):
                print("Tin hieu la nhan dc la: ", msg) 
                phat_loa_7_dauhieu_sot()
                send_message_to_arduino(arduino_module,"raspi_da_doc_canhbao")
                temp_C = int(msg[-4:])/100
                step = 4

    while step ==4:
        ## Load excel file
        data = pd.read_excel(csv_path)
        qldt_row.append(text_split[0])
        qldt_row.append(text_split[1])
        qldt_row.append(str(temp_C))
        qldt_row.append(text_split[2])
        qldt_row.append(face_path)
        print(qldt_row)
        ##Save to CSV file
        input_data = pd.Series(qldt_row, index=header)
        data = data.append(input_data, ignore_index=True)
        print(data)
        # data.to_excel(csv_path,engine='xlsxwriter')
        writer = pd.ExcelWriter(csv_path, engine='xlsxwriter')
        data.to_excel(writer, index=False)
        writer.save()

        step =5
    while step ==5:
        if arduino_module.in_waiting > 0:
            msg = read_message_from_arduino(arduino_module)
            if (msg == "ard_finish"):
                visualize_cam_on()
                print("Tin hieu la nhan dc la: ", msg) 
                step = 1
                time.sleep(1)
                qldt_row.clear()        

    key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
face_cam.stop()