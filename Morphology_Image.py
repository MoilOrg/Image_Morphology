import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
from imutils import contours
from imutils import perspective
from scipy.spatial.distance import euclidean

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def show_images(images): 
    for i, image in enumerate(images):
        cv2.imshow("Original" + str(i), image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

img = ("geometri.png")
read = cv2.imread(img)
read = cv2.resize(read, (600, 400), interpolation=cv2.INTER_AREA)
_, mask = cv2.threshold(read, 220, 255, cv2.THRESH_BINARY_INV)
kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))

gray = cv2.cvtColor(read, cv2.IMREAD_GRAYSCALE)
morh = cv2.cvtColor(read, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7, 7), 0)

edged = cv2.Canny(blur, 100, 200)
edged = cv2.dilate(edged, kernal, iterations=1)
edged = cv2.erode(edged, kernal, iterations=1)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)

counts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
counts = imutils.grab_contours(counts)

print("Total number of contours are: ", len(counts))

(counts, _) = contours.sort_contours(counts)
counts = [x for x in counts if cv2.contourArea(x) > 500]

objects =counts[0]
border = cv2.minAreaRect(objects)
border = cv2.boxPoints(border)
border = np.array(border, dtype="int")
border = perspective.order_points(border)
(tl, tr, br, bl) = border
dist_in_pixel = euclidean(tl, tr)
dist_in_cm = 2
pixel_per_cm = dist_in_pixel / dist_in_cm

for count in counts:
    border = cv2.minAreaRect(count)
    border = cv2.boxPoints(border)
    border = np.array(border, dtype="int")
    border = perspective.order_points(border)
    (tl, tr, br, bl) = border
    cv2.drawContours(morh, [border.astype("int")], -1, (0, 0, 255), 2)
    mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0]) / 2), tl[1] + int(abs(tr[1] - tl[1]) / 2))
    mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0]) / 2), tr[1] + int(abs(tr[1] - br[1]) / 2))
    width  = euclidean(tl, tr)
    length = euclidean(tr, br)

    cv2.line(border, (0,0), (150,150), (255,255,255), 15)
    cv2.rectangle(border, (15,25), (200, 150), (0,255,0), 5)
    cv2.circle(border, (100,63), 55, (0,0,255), -1)
    cv2.putText(morh, "{:.1f} px".format(width), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.putText(morh , "{:.1f} px".format(length), (int(mid_pt_verticle[0] + 10), int(mid_pt_verticle[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2) 

    print("Total contours processed: ", counts)

    plt.subplot(121),plt.imshow(read),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
    plt.xticks([]), plt.yticks([])
    cv2.imshow("Egded", edged)
    cv2.imshow("morh", morh)
    cv2.imshow("opening", opening)
    cv2.waitKey(0)
    show_images([read])












