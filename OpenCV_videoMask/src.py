import numpy as np
import cv2 as cv
path = r"C:\Users\HABIB\Documents\MCS\dips\victim.mp4"
samplepath = r"C:\Users\HABIB\Documents\MCS\dips\sample.jpeg"
#maskpath = r"C:\Users\HABIB\Documents\MCS\dips\masked.png"
maskpath2 = r"C:\Users\HABIB\Documents\MCS\dips\test2.jpeg"
outpath = r"C:\Users\HABIB\Documents\MCS\dips\output.mp4"
cap = cv.VideoCapture(path)
for i in range (0,200):
    ret , sample = cap.read()
    if i == 40:
        cv.imwrite(samplepath,sample)
        break
cap.release()
cv.destroyAllWindows()

def process(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blur = cv.GaussianBlur(img_gray, (3, 3), 0)
    img_canny = cv.Canny(img_blur, 161, 54)
    img_dilate = cv.dilate(img_canny, None, iterations=1)
    return cv.erode(img_dilate, None, iterations=1)

def get_watermark(img):
    contours, _ = cv.findContours(process(img), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    img.fill(255)
    for cnt in contours:
        if cv.contourArea(cnt) > 100:
            cv.drawContours(img, [cnt], -1, 0, -1)

img = cv.imread(r"C:\Users\HABIB\Documents\MCS\dips\sample.jpeg")
size =  (360,640)
get_watermark(img)

cv.destroyAllWindows()
result = cv.VideoWriter(outpath, 
                         cv.VideoWriter_fourcc(*'mp4v'),
                         20, size)
cap = cv.VideoCapture(path)
while True:
    ret, frame = cap.read()
    mask= cv.imread(maskpath2,0)
    masked = cv.inpaint(frame,mask, 3, cv.INPAINT_TELEA)
    masked = cv.resize(masked, (360,640), cv.INTER_AREA)
    n=0
    n+=1
    if n == 200:
        break
    else:
        result.write(masked)
        cv.imshow("RESLT",masked)
        if cv.waitKey(1)==27:
            break
cap.release()
result.release()
cv.destroyAllWindows()
print("DONE")

