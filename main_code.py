from imutils import face_utils 
import dlib
import cv2
import pyautogui as pag
import numpy as n
import time

def Mar(p1,p2,p3,p4,p5,p6,p7,p8):
	mar = (dst(p1,p2) + dst(p3,p4) + dst(p5,p6))/(3.0*dst(p7,p8))
	return mar

def Ear(p1,p2,p3,p4,p5,p6):
	ear = (dst(p2,p6) + dst(p3,p5))/(2*dst(p1,p4))*1.0
	return ear
def dst(p1,p2):
	dist = n.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
	return dist

def angle(p1):
	slp = (p1[1] - 250)/(p1[0] - 250)*1.0
	agle = 1.0*n.arctan(slp)
	return agle


leclick = n.array([])
riclick = n.array([])
scroll= n.array([])
eye__open = n.array([])
leclickarea = n.array([])
riclickarea = n.array([])
pag.PAUSE =0
y= "shape_predictor_68_face_landmarks.dat"
det = dlib.get_frontal_face_detector() 
pred= dlib.shape_predictor(y) 
(l1start,l1end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(r1start,r1end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(m1start,m1end) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

cp= cv2.VideoCapture(0)
ft = cv2.FONT_HERSHEY_SIMPLEX
ctime = time.time()
while(time.time() - ctime <= 25):
	r,img = cp.read()
	bimg = n.zeros((480,640,3),dtype = n.uint8)
	img = cv2.flip(img,1)
	gry = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	cl = cv2.createCLAHE(clipLimit = 2.1, tileGridSize = (8,8))
	gry = cl.apply(gry)
	rets  = det(gry,0)
	for (i,ret) in enumerate(rets ): 
		shp = pred(gry,ret)
		shp = face_utils.shape_to_np(shp)
		leye = Ear(shp[36],shp[37],shp[38],shp[39],shp[40],shp[41])
		reye = Ear(shp[42],shp[43],shp[44],shp[45],shp[46],shp[47])
		mar = Mar(shp[50],shp[58],shp[51],shp[57],shp[52],shp[56],shp[48],shp[54]) 
		EYE_DIFF = (leye - reye)*100
		ll__Eye = shp[l1start:l1end]
		rr__Eye = shp[r1start:r1end]
		mroi = shp[m1start:m1end]
		lEyehull = cv2.convexHull(ll__Eye)
		rEyehull = cv2.convexHull(rr__Eye)
		mthull = cv2.convexHull(mroi)
		cv2.drawContours(img,[mroi],-1,(0,255,60),1)
		cv2.drawContours(img,[lEyehull],-1,(0,255,60),1)
		cv2.drawContours(img,[rEyehull],-1,(0,255,60),1)
		Mouarea = cv2.contourArea(mthull)
		lefarea = cv2.contourArea(lEyehull)
		Rigarea = cv2.contourArea(rEyehull)
		etime = time.time() - ctime
		if etime < 5.0: 
			cv2.putText(bimg,'Open Eyes',(0,100), ft, 1,(245,245,245),2,cv2.LINE_AA)
			eye__open = n.append(eye__open,[EYE_DIFF])
		elif etime > 5.0 and etime < 10.0:
			cv2.putText(bimg,'left eye close',(0,100), ft, 1,(245,245,245),2,cv2.LINE_AA)
			leclick = n.append(leclick,[EYE_DIFF])
			leclickarea = n.append(leclickarea,[lefarea])
		elif etime > 11.0 and etime < 18.0: 
			cv2.putText(bimg,'close right eye and open left eye',(0,100), ft, 1,(245,245,245),2,cv2.LINE_AA)
			riclick = n.append(riclick,[EYE_DIFF])
			riclickarea = n.append(riclickarea,[Rigarea])
		elif etime > 19.0 and etime < 24.0: 
			cv2.putText(bimg,'Mouth open',(0,100),ft,1,(245,245,245),2,cv2.LINE_AA)
			if etime > 21.0 and etime < 24.0:
				scroll = n.append(scroll,[mar])
		else: 
			pass 
		for (x,y) in shp: 
			cv2.circle(img,(x,y),2,(0,255,60),-1)
		out = n.vstack((img,bimg))
		cv2.imshow('CALIBRATION:',out)
	if cv2.waitKey(5) & 0xff == 113: 
		break
cp.release()
cv2.destroyAllWindows()

cp = cv2.VideoCapture(0)
MAR = n.array([])
sc_sts = 0 
eye__open = n.sort(eye__open)
leclick = n.sort(leclick)
riclick = n.sort(riclick)
scroll = n.sort(scroll)
leclickarea = n.sort(leclickarea)
riclickarea = n.sort(riclickarea)
openeyes = n.median(eye__open) 
LEFT_CLICK = n.median(leclick) - 1 
RIGHT_CLICK = n.median(riclick) + 1 
scrllng = n.median(scroll)
LEFTCLICK_AREA = n.median(leclickarea)
RIGHTCLICK_AREA = n.median(riclickarea)

while(True):
	try: 
		bimg = n.zeros((480,640,3),dtype = n.uint8)
		__, img = cp.read() 
		img=cv2.flip(img,1)
		gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		cl = cv2.createCLAHE(clipLimit = 2.1, tileGridSize = (8,8))
		gry = cl.apply(gry)
		cv2.circle(img,(250,250),50,(0,0,255),2)
	    
		rets = det(gry, 0)
	    
		for (i, ret) in enumerate(rets ):
		
			shp = pred(gry, ret)
			shp = face_utils.shape_to_np(shp)
			[h,k] = shp[33]
			leye = Ear(shp[36],shp[37],shp[38],shp[39],shp[40],shp[41])
			reye =  Ear(shp[42],shp[43],shp[44],shp[45],shp[46],shp[47])
			EYE_DIFF = (leye - reye )*100 
			mar = Mar(shp[50],shp[58],shp[51],shp[57],shp[52],shp[56],shp[48],shp[54]) 			
			cv2.line(img,(250,250),(h,k),(0,0,0),1)
			ll__Eye = shp[l1start:l1end]
			rr__Eye = shp[r1start:r1end]
			mroi = shp[m1start:m1end]
			lEyehull = cv2.convexHull(ll__Eye)
			rEyehull = cv2.convexHull(rr__Eye)
			mthull = cv2.convexHull(mroi)
			cv2.drawContours(img,[mroi],-1,(0,255,60),1) 
			cv2.drawContours(img,[lEyehull],-1,(0,255,60),1)
			cv2.drawContours(img,[rEyehull],-1,(0,255,60),1)
			lefarea = cv2.contourArea(lEyehull)
			Rigarea = cv2.contourArea(rEyehull)
			Mouarea = cv2.contourArea(mthull)
			if EYE_DIFF < LEFT_CLICK and lefarea < LEFTCLICK_AREA: 
				pag.click(button = 'Left')
				cv2.putText(bimg,"LEFTCLICK",(0,100),ft,1,(245,245,245),2,cv2.LINE_AA)
				leclick = n.array([])
			elif EYE_DIFF > RIGHT_CLICK and Rigarea < RIGHTCLICK_AREA:  
				pag.click(button = 'Right') 
				cv2.putText(bimg,"RIGHTCLICK",(0,100),ft,1,(245,245,245),2,cv2.LINE_AA)
				leclick = n.array([])
		
		for (x, y) in shp:
			cv2.circle(img, (x, y), 2, (0, 245, 1), -1)
		MAR = n.append(MAR,[mar]) 
		if len(MAR) == 30: 
			mar_avg = n.mean(MAR)
			MAR = n.array([]) 
			if int(mar_avg*100) > int(scrllng*100):
				if sc_sts == 0:
					sc_sts = 1
				else:
					sc_sts = 0
		
		if sc_sts == 0:
			if((h-250)**2 + (k-250)**2 - 50**2 > 0):
				a = angle(shp[33])
				if h > 250: 
					time.sleep(0.03)
					x = 10*n.cos(1.0*a)
					y = 10*n.sin(1.0*a)
					pag.moveTo(pag.position()[0]+(x),pag.position()[1]+(y),duration = 0.01)
					cv2.putText(bimg,"Cursor in motion",(0,150),ft,1,(245,245,245),2,cv2.LINE_AA)
				else:
					time.sleep(0.03)
					x = 10*n.cos(1.0*a)
					y = 10*n.sin(1.0*a)
					pag.moveTo(pag.position()[0]-(x),pag.position()[1]-(y),duration = 0.01)
					cv2.putText(bimg,"Cursor in motion",(0,150),ft,1,(245,245,245),2,cv2.LINE_AA)
		else: 
			cv2.putText(bimg,'Scrolling mode is Activated',(0,100),ft,1,(245,245,245),2,cv2.LINE_AA)
			if k > 300:
				cv2.putText(bimg,"Down ",(0,150),ft,1,(245,245,245),2,cv2.LINE_AA)
				pag.scroll(-12)
			elif k < 200:
				cv2.putText(bimg,"Up  ",(0,150),ft,1,(245,245,245),2,cv2.LINE_AA) 
				pag.scroll(12)
			else:
				pass
		cv2.circle(img,(h,k),2,(245,1,0),-1)
		cv2.putText(bimg,"Press the key q to terminate",(0,200),ft,1,(0,10,245),2,cv2.LINE_AA)
		out = n.vstack((img,bimg))
		cv2.imshow('Control cursor',out)
		r = cv2.waitKey(5) & 0xFF
		if r == 113:
			break
	except: 
		bimg = n.zeros((480,640,3),dtype = n.uint8)
		cv2.putText(bimg,"No Landmark is detected",(0,100),ft,1,(0,10,245),2,cv2.LINE_AA)
		__,img = cp.read() 
		img= cv2.flip(img,1)
		res = n.vstack((img,bimg))
		cv2.imshow('Cursor Control',res)
		r = cv2.waitKey(5) & 0xff
		if r == 113:
			break

cv2.destroyAllWindows()
cp.release()
