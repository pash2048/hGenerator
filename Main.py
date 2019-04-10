import numpy as np
import cv2
import pandas as pd
import random
import imutils

def imageName(numberFream):
    return "IMG-"+str(numberFream)+".jpg"

def add_to_dataframe(data,numberFream,x,y,bw,bh,c):
    name = imageName(numberFream)
    bx = x + bw/2
    by = y + bh/2
    return data.append({'NAME' : name, 'BX' : bx, 'BY' : by, 'BW' : bw , 'BH' : bh , 'C' : c} , ignore_index=True)

def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
	bg_img = background_img.copy()
	
	if overlay_size is not None:
		img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

	b,g,r,a = cv2.split(img_to_overlay_t)
	overlay_color = cv2.merge((b,g,r))
	
	mask = cv2.medianBlur(a,5)

	h, w, _ = overlay_color.shape
	roi = bg_img[y:y+h, x:x+w]
	img1_bg = cv2.bitwise_and(roi.copy(),roi.copy(),mask = cv2.bitwise_not(mask))
	img2_fg = cv2.bitwise_and(overlay_color,overlay_color,mask = mask)

	bg_img[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)

	return bg_img

data = pd.DataFrame(columns=['NAME', 'BX', 'BY','BW','BH','C'])

overlay_t = cv2.imread('H.png',-1) 
cap = cv2.VideoCapture('1.mp4')

bh, bw, channelT = overlay_t.shape

w = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    img = frame
    x = 0
    y = 0
    if not ret:
        break
    q = bool(random.getrandbits(1))
    if q:
        height, width, channel = frame.shape
        x = random.randint(0,width-bw)
        y = random.randint(0,height-bh)
        #stream = imutils.rotate(overlay_t, random.randint(1,100))
        img = overlay_transparent(frame, overlay_t, x, y)
    cv2.imshow('image',img)
    data = add_to_dataframe(data,w,x,y,bw,bh,q)
    cv2.imwrite("img/"+imageName(w),img)
    w+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

data.to_csv("dataset.csv")



