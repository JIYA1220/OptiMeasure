import cv2
import numpy as np
import utils

# --- CONFIGURATION ---
source = '1.jpeg' 
image_path = '1.jpeg'
# ---------------------

def empty(a):
    pass

# 1. Setup Windows and Layout (Smaller and more compact)
cv2.namedWindow("Settings")
cv2.moveWindow("Settings", 0, 0)
cv2.resizeWindow("Settings", 450, 350)

cv2.namedWindow("Original Video Source")
cv2.moveWindow("Original Video Source", 460, 0)

cv2.namedWindow("A4 Paper Measurements")
cv2.moveWindow("A4 Paper Measurements", 460, 420)

# Sliders
cv2.createTrackbar("Paper Thr1", "Settings", 100, 255, empty)
cv2.createTrackbar("Paper Thr2", "Settings", 100, 255, empty)
cv2.createTrackbar("Paper MinArea", "Settings", 10000, 100000, empty)
cv2.createTrackbar("Obj Thr1", "Settings", 50, 255, empty)
cv2.createTrackbar("Obj Thr2", "Settings", 50, 255, empty)
cv2.createTrackbar("Obj MinArea", "Settings", 2000, 20000, empty)
cv2.createTrackbar("Show All Shapes", "Settings", 0, 1, empty)

is_dynamic = isinstance(source, int) or (isinstance(source, str) and (source.endswith('.mp4') or source.endswith('.avi')))

cap = None
if is_dynamic:
    cap = cv2.VideoCapture(source)

scaleFactor = 3
wPaper = 210 * scaleFactor
hPaper = 297 * scaleFactor

while True:
    pT1 = cv2.getTrackbarPos("Paper Thr1", "Settings")
    pT2 = cv2.getTrackbarPos("Paper Thr2", "Settings")
    pArea = cv2.getTrackbarPos("Paper MinArea", "Settings")
    oT1 = cv2.getTrackbarPos("Obj Thr1", "Settings")
    oT2 = cv2.getTrackbarPos("Obj Thr2", "Settings")
    oArea = cv2.getTrackbarPos("Obj MinArea", "Settings")
    showAll = cv2.getTrackbarPos("Show All Shapes", "Settings")

    if is_dynamic:
        success, img = cap.read()
        if not success:
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else: break
    else:
        img = cv2.imread(image_path)
        if img is None: break

    imgOriginal = img.copy()

    # 2. Find the A4 Paper
    imgContours, finalCountours = utils.getContours(img, minArea=pArea, filter=4 if showAll==0 else 0, 
                                                   cannyThreshold=[pT1, pT2], draw=True if showAll==1 else False)

    if len(finalCountours) != 0:
        biggest = finalCountours[0][2]
        cv2.drawContours(imgOriginal, [biggest], -1, (0, 255, 0), 10)
        cv2.putText(imgOriginal, "PAPER FOUND", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
        
        imgWarp = utils.warpImg(img, biggest, wPaper, hPaper)
        imgWarpDraw, finalCountours2 = utils.getContours(imgWarp, minArea=oArea, filter=0, 
                                                            cannyThreshold=[oT1, oT2], draw=False)
        
        if len(finalCountours2) != 0:
            for obj in finalCountours2:
                cv2.drawContours(imgWarpDraw, [obj[2]], -1, (255, 0, 0), 2)
                approx = obj[2]
                newPoints = utils.reorder(approx) if len(approx) == 4 else None
                x, y, w, h = obj[3]
                
                if newPoints is not None:
                    newWidth = round(utils.findDistance(newPoints[0][0]//scaleFactor, newPoints[1][0]//scaleFactor) / 10, 1)
                    newHeight = round(utils.findDistance(newPoints[0][0]//scaleFactor, newPoints[2][0]//scaleFactor) / 10, 1)
                    cv2.arrowedLine(imgWarpDraw, (newPoints[0][0][0], newPoints[0][0][1]), (newPoints[1][0][0], newPoints[1][0][1]), (255, 0, 255), 3, 8, 0, 0.05)
                    cv2.arrowedLine(imgWarpDraw, (newPoints[0][0][0], newPoints[0][0][1]), (newPoints[2][0][0], newPoints[2][0][1]), (255, 0, 255), 3, 8, 0, 0.05)
                else:
                    newWidth = round((w / scaleFactor) / 10, 1)
                    newHeight = round((h / scaleFactor) / 10, 1)
                
                cv2.putText(imgWarpDraw, f'{newWidth}cm', (x + 10, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (255, 0, 255), 2)
                cv2.putText(imgWarpDraw, f'{newHeight}cm', (x - 60, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (255, 0, 255), 2)

        cv2.imshow('A4 Paper Measurements', cv2.resize(imgWarpDraw, (250, 350)))
    else:
        cv2.putText(imgOriginal, "NO PAPER", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
        cv2.imshow('A4 Paper Measurements', np.zeros((100, 250, 3), np.uint8))

    # 3. Resize Original Video (Strict limit for small screens)
    h_orig, w_orig, _ = imgOriginal.shape
    aspect = w_orig / h_orig
    new_w = 400
    new_h = int(new_w / aspect)
    # If it's too tall, limit height instead
    if new_h > 400:
        new_h = 400
        new_w = int(new_h * aspect)
        
    display_img = cv2.resize(imgOriginal if showAll==0 else imgContours, (new_w, new_h))
    cv2.imshow('Original Video Source', display_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if cap:
    cap.release()
cv2.destroyAllWindows()
