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
mixer.init()

# Grab path to current working directory
CWD_PATH = os.getcwd()
audio_path = os.path.join(CWD_PATH,"Audio_QLDT")
out_img_path = os.path.join(CWD_PATH,"Out_Img")
csv_path = os.path.join(CWD_PATH,"System_1.xlsx")
sys_img_path = os.path.join(CWD_PATH,"System_images")


face_show_location = [0,0]
system_name_location = [640,0]
info_show_location = [640,480]
def showInFixedWindow(winname, img, x, y):
    cv2.namedWindow(winname, flags=cv2.WINDOW_GUI_NORMAL+ cv2.WINDOW_AUTOSIZE)        # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    cv2.imshow(winname,img)

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

def phat_loa_1_dung_2banchan():
    visualize_dung_2banchan()
    phat_loa("1_dung_2banchan.wav")
    # phat_loa_untill_end("1_dung_2banchan.wav")
    print("1. Da phat loa chao mung va dung vao vi tri 2 ban chan")
    time.sleep(0.5)

def visualize(face_frame):
    showInFixedWindow("Face_Camera", face_frame,face_show_location[0],face_show_location[1])

def visualize_ten_hethong():
    ## 0. Load hinh ten he thong
    img = cv2.imread(os.path.join(sys_img_path,"ten_hethong.png"))
    img = imutils.resize(img, width = 320)
    showInFixedWindow('System Name',img, system_name_location[0], system_name_location[1])
    cv2.waitKey(1)

def visualize_dung_2banchan():
    ## 1. Load hinh dung 2 ban chan
    img = cv2.imread(os.path.join(sys_img_path,"dung_2banchan.png"))
    img = imutils.resize(img, width = 320)
    showInFixedWindow('System Notice 1',img, info_show_location[0], info_show_location[1])
    cv2.waitKey(1)

def visualize_deo_khautrang():
    ## 2. Load hinh deo khau trang
    img = cv2.imread(os.path.join(sys_img_path,"deo_khautrang.png"))
    img = imutils.resize(img, width = 320)
    showInFixedWindow('System Notice 2',img, info_show_location[0], info_show_location[1])
    cv2.waitKey(1)

def visualize_dua_maQR():
    ## 3. Load hinh dua ma QR
    img = cv2.imread(os.path.join(sys_img_path,"dua_maQR.png"))
    img = imutils.resize(img, width = 320)
    showInFixedWindow('System Notice 3',img, info_show_location[0], info_show_location[1])
    cv2.waitKey(1)

def visualize_dua_dt_gan():
    ## 4. Load hinh dua dien thoai gan
    img = cv2.imread(os.path.join(sys_img_path,"dua_dt_gan.png"))
    img = imutils.resize(img, width = 320)
    showInFixedWindow('System Notice 4',img, info_show_location[0], info_show_location[1])
    cv2.waitKey(1)

def visualize_taolai_maQR():
    ## 4.0 Load hinh tao lai ma QR
    img = cv2.imread(os.path.join(sys_img_path,"taolai_maQR.png"))
    img = imutils.resize(img, width = 320)
    showInFixedWindow('System Notice 4.0',img, info_show_location[0], info_show_location[1])
    cv2.waitKey(1)

def visualize_diqua_vungdo():
    ## 5. Load hinh di qua vung do
    img = cv2.imread(os.path.join(sys_img_path,"diqua_vungdo.png"))
    img = imutils.resize(img, width = 320)
    showInFixedWindow('System Notice 5',img, info_show_location[0], info_show_location[1])
    cv2.waitKey(1)

def visualize_nhin_thang_camera():
    ## 6. Load hinh nhin thang camera
    img = cv2.imread(os.path.join(sys_img_path,"nhin_thang_camera.png"))
    img = imutils.resize(img, width = 320)
    showInFixedWindow('System Notice 6',img, info_show_location[0], info_show_location[1])
    cv2.waitKey(1)

def visualize_do_nhietdo():
    ## 7. Load hinh do nhiet do
    img = cv2.imread(os.path.join(sys_img_path,"do_nhietdo.png"))
    img = imutils.resize(img, width = 320)
    showInFixedWindow('System Notice 7',img, info_show_location[0], info_show_location[1])
    cv2.waitKey(1)

def visualize_nhietdo_binhthuong():
    ## 8. Load hinh nhiet do binh thuong
    img = cv2.imread(os.path.join(sys_img_path,"nhietdo_binhthuong.png"))
    img = imutils.resize(img, width = 320)
    showInFixedWindow('System Notice 8',img, info_show_location[0], info_show_location[1])
    cv2.waitKey(1)

def visualize_dauhieu_sot():
    ## 9. Load hinh dau hieu sot
    img = cv2.imread(os.path.join(sys_img_path,"dauhieu_sot.png"))
    img = imutils.resize(img, width = 320)
    showInFixedWindow('System Notice 9',img, info_show_location[0], info_show_location[1])
    cv2.waitKey(1)

def visualize_cam_on():
    ## 10. Load hinh cam on
    img = cv2.imread(os.path.join(sys_img_path,"cam_on.png"))
    img = imutils.resize(img, width = 320)
    showInFixedWindow('System Notice 10',img, info_show_location[0], info_show_location[1])
    cv2.waitKey(1)

while True:
    # phat_loa_1_dung_2banchan()
    visualize_ten_hethong()
    visualize_cam_on()
    # check to see if a key was pressed
    key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
print("[INFO] cleaning up...")
cv2.destroyAllWindows()