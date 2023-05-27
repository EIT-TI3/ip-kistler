import numpy as np
import cv2 as cv
import glob

DATADIR = './data'
CB_SHAPE = (9,6)

with np.load(DATADIR + '/data.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

def draw_axis(img, corners, imgpts:np.ndarray):
    corner = tuple(np.asarray(corners[0], dtype=np.int16).ravel())
    img = cv.line(img, corner, tuple(np.asarray(imgpts[0], dtype=np.int16).ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(np.asarray(imgpts[1], dtype=np.int16).ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(np.asarray(imgpts[2], dtype=np.int16).ravel()), (0,0,255), 5)
    return img

def draw_cube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((CB_SHAPE[0]*CB_SHAPE[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CB_SHAPE[0],0:CB_SHAPE[1]].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
axis_c = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

cap = cv.VideoCapture(2)
if not cap.isOpened():
    print("Cannot open camera")
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    img = frame
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, CB_SHAPE,None)
    if ret == True:
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis_c, rvecs, tvecs, mtx, dist)
        img = draw_cube(img,corners2,imgpts)

    cv.imshow('img',img)

    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()
