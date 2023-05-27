import numpy as np
import cv2 as cv
import glob


DATADIR = './data'
CB_SHAPE = (9,6)


cap = cv.VideoCapture(2)
if not cap.isOpened():
    print("Cannot open camera")
    exit()


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((CB_SHAPE[0]*CB_SHAPE[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CB_SHAPE[0],0:CB_SHAPE[1]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


def get_error(objpoints, rvecs, imgpoints, tvecs, mtx, dist):
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "total error: {}".format(mean_error/len(objpoints)) )


def undistort(img, mtx, dist):
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imshow('undistorted', dst)
    cv.imshow('original', img)


while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    img = frame
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, CB_SHAPE, None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        
        # Draw and display the corners
        cv.drawChessboardCorners(frame, CB_SHAPE, corners2, ret)

        if cv.waitKey(1) == ord('c'):
            objpoints.append(objp)
            imgpoints.append(corners2)
            ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            
            get_error(objpoints, rvecs, imgpoints, tvecs, mtx, dist)
            #undistort(frame, mtx, dist)

    cv.imshow('img', frame)

    if cv.waitKey(1) == ord('q'):
        np.savez(DATADIR + '/data.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
        np.save(DATADIR + '/imgpoints.np', imgpoints)
        np.save(DATADIR + '/objpoints.np', objpoints)
        break

cv.destroyAllWindows()
 