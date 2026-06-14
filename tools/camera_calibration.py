import numpy as np
import cv2
import glob
import os

def calibrate_fisheye(showPics=True):
    root = os.getcwd()
    calibrationDir = os.path.join(root, "calibration_images")
    print(calibrationDir)
    imgPathList = glob.glob(os.path.join(calibrationDir, '*.png'))

    # Chessboard inner corners
    nCols = 8  # inner corners along width
    nRows = 6   # inner corners along height
    squareSize = 0.0286  # meters (~28.6 mm squares)

    termCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    # Prepare object points
    objp = np.zeros((nRows*nCols,3), np.float32)
    objp[:,:2] = np.mgrid[0:nCols, 0:nRows].T.reshape(-1,2)
    objp *= squareSize  # scale to real size

    objPoints = []
    imgPoints = []

    for imgPath in imgPathList:
        img = cv2.imread(imgPath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(gray, (nCols, nRows), None)

        if found:
            objPoints.append(objp.reshape(-1,1,3).astype(np.float64))
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), termCriteria)
            imgPoints.append(corners2.astype(np.float64))

            if showPics:
                cv2.drawChessboardCorners(img, (nCols,nRows), corners2, found)
                cv2.imshow("Corners", img)
                cv2.waitKey(500)
    cv2.destroyAllWindows()

    # Fisheye calibration
    K = np.zeros((3,3))
    D = np.zeros((4,1))
    rvecs = []
    tvecs = []

    N_OK = len(objPoints)
    print(f"Calibrating with {N_OK} valid images...")

    ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objPoints,
        imgPoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
        termCriteria
    )

    print("Camera Matrix K:\n", K)
    print("Distortion Coefficients D:\n", D.ravel())
    print(f"Reprojection Error: {ret:.4f}")

    paramPath = os.path.join(root, "fishEye_RB3_CameraCalibration_fisheye.npz")
    np.savez(paramPath, repError=ret, camMatrix=K, distCoeff=D, rvecs=rvecs, tvecs=tvecs)

    return K, D, rvecs, tvecs


def undistort_fisheye_image(img, K, D):
    h, w = img.shape[:2]
    newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w,h), np.eye(3), balance=0.0)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), newK, (w,h), cv2.CV_16SC2)
    undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
    return undistorted


if __name__ == "__main__":
    K, D, rvecs, tvecs = calibrate_fisheye(showPics=False)

    # Example test image
    testPath = os.path.join(os.getcwd(), "calibration_images", "calib_011.png")
    img = cv2.imread(testPath)
    undistorted = undistort_fisheye_image(img, K, D)

    img = img[::2,::2,:]  # downsample for display
    undistorted = undistorted[::2,::2,:]

    cv2.imshow("Original", img)
    cv2.imshow("Undistorted", undistorted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # calibrate_fisheye(True)