import math
from tkinter.filedialog import askopenfilename

import cv2
import numpy as np

lastCoord = []
regionSelected = False
mouseClicked = False
actualPixelHSV = []
diff_H = 5
diff_S = 50
diff_V = 50

# Values for HSV visualization
# Size of the result image in pixels
HSV_SIZE_X = 256
HSV_SIZE_Y = HSV_SIZE_X
# Geometry values
HSV_SIZE_HALF = HSV_SIZE_X >> 1
HSV_CENTER_X = HSV_SIZE_HALF
HSV_CENTER_Y = HSV_SIZE_HALF
HSV_VALUE_SLIDER_HEIGHT = 30
HSV_V_VALUE_TICK_HEIGHT_HALF = 3
HSV_V_VALUE_TICK_BOUND_HEIGHT_HALF = 5
# To make sure the maximal distance of the circle gets value 255
HSV_FACTOR = 255 / HSV_SIZE_HALF
# Default V value
HSV_DEFAULT_V_VALUE = 192

#normalizált ablakok hogy nagy képek is ráférjenek a képernyőre
cv2.namedWindow("resized_original", cv2.WINDOW_NORMAL)
cv2.namedWindow("resized_segmented", cv2.WINDOW_NORMAL)


def draw_HS_line(img, Hue, Sat, color, drawCircle = False):
    angle = Hue * 2
    R = HSV_SIZE_HALF * Sat / 255
    destX = int(HSV_CENTER_X + R * math.cos(math.radians(angle)))
    destY = HSV_SIZE_Y - int(HSV_CENTER_Y + R * math.sin(math.radians(angle)))
    cv2.line(img, (HSV_CENTER_X, HSV_CENTER_Y), (destX, destY), color, 1)
    if drawCircle:
        cv2.circle(img, (destX, destY), 3, color, -1)


def draw_V_tick(img, Val, hh, color, thick=1):
    tick_y = HSV_SIZE_Y + (HSV_VALUE_SLIDER_HEIGHT >> 1)
    tick_x = int(HSV_SIZE_X * Val / 255.0)
    cv2.line(img, (tick_x, tick_y - hh), (tick_x, tick_y + hh), color, thick)


def draw_Saturation_circle(img, Sat, color):
    R = int(HSV_SIZE_HALF * Sat / 255)
    cv2.circle(img, (HSV_CENTER_X, HSV_CENTER_Y), R, color)


def update_HSV_palette(value_V):
    global hsv_palette_circle, hsv_palette_circle_mask, hsv_palette_bgr

    hsv_palette_circle[hsv_palette_circle_mask > 0] = value_V
    hsv_palette_bgr = cv2.cvtColor(hsv_palette_circle, cv2.COLOR_HSV2BGR)
    tick_y = HSV_SIZE_Y + (HSV_VALUE_SLIDER_HEIGHT >> 1)
    cv2.line(hsv_palette_bgr, (0, tick_y), (HSV_SIZE_X, tick_y), (0, 0, 0), 1)
    draw_V_tick(hsv_palette_bgr, value_V, HSV_V_VALUE_TICK_HEIGHT_HALF, (0, 0, 0), 3)


def visualize_hsv_segment_parameters(pixel_HSV, min_H, max_H, min_S, max_S, min_V, max_V):
    # HSV vizualizacio
    update_HSV_palette(pixel_HSV[2])
    draw_HS_line(hsv_palette_bgr, pixel_HSV[0], pixel_HSV[1], (0, 0, 0), True)
    draw_HS_line(hsv_palette_bgr, min_H, 255, (255, 255, 255))
    draw_HS_line(hsv_palette_bgr, max_H, 255, (255, 255, 255))
    draw_Saturation_circle(hsv_palette_bgr, min_S, (0, 255, 0))
    draw_Saturation_circle(hsv_palette_bgr, max_S, (0, 0, 255))
    draw_V_tick(hsv_palette_bgr, min_V, HSV_V_VALUE_TICK_BOUND_HEIGHT_HALF, (0, 192, 0), 3)
    draw_V_tick(hsv_palette_bgr, max_V, HSV_V_VALUE_TICK_BOUND_HEIGHT_HALF, (0, 0, 192), 3)
    draw_V_tick(hsv_palette_bgr, pixel_HSV[2], HSV_V_VALUE_TICK_HEIGHT_HALF, (0, 0, 0), 3)
    if(min_H > max_H):
        min_H = min_H - 180
    print('H: [', min_H, '-', max_H, ']; S: [', min_S, '-', max_S, ']; V: [', min_V, '-', max_V, ']')
    cv2.imshow('palette', hsv_palette_bgr)


def segmentHSVPoint():
    # Globalis valtozok atvetele
    global img, imgHSV, hsv_palette_circle
    global lastCoord, regionSelected, mouseClicked
    global diff_H, diff_S, diff_V
    global actualPixelHSV
    global segmented

    # Ha nem volt elozo kattintas
    if len(actualPixelHSV) == 0:
        return

    pixel_HSV = actualPixelHSV

    # Alulcsordulas kezelese a szegmentalasi reszben
    min_H = pixel_HSV[0] - diff_H

    # Tulcsordulas kezelese a szegmentalasi reszben
    max_H = pixel_HSV[0] + diff_H

    if pixel_HSV[1] > diff_S:
        min_S = pixel_HSV[1] - diff_S
    else:
        min_S = 0

    if pixel_HSV[1] < (255 - diff_S):
        max_S = pixel_HSV[1] + diff_S
    else:
        max_S = 255

    if pixel_HSV[2] > diff_V:
        min_V = pixel_HSV[2] - diff_V
    else:
        min_V = 0

    if pixel_HSV[2] < (255 - diff_V):
        max_V = pixel_HSV[2] + diff_V
    else:
        max_V = 255

    # HSV intervallum szegmentalas
    # H ertek alul- es tulcsordulasanak kezelesevel: szukseg eseten 2 intervallumos szegmentalas kell
    vis_min_H = min_H
    vis_max_H = max_H
    segmented = np.zeros(imgHSV.shape[0:2], np.uint8)
    if min_H < 0:
        while min_H < 0:
            min_H = min_H + 180
        minHSV = np.array([min_H, min_S, min_V])
        maxHSV = np.array([180, max_S, max_V])
        segmented = cv2.inRange(imgHSV, minHSV, maxHSV)
        vis_min_H = min_H
        min_H = 0

    if max_H > 180:
        while max_H > 180:
            max_H = max_H - 180
        minHSV = np.array([0, min_S, min_V])
        maxHSV = np.array([max_H, max_S, max_V])
        segmented_temp = cv2.inRange(imgHSV, minHSV, maxHSV)
        segmented = cv2.bitwise_or(segmented, segmented_temp)
        vis_max_H = max_H
        max_H = 180

    minHSV = np.array([min_H, min_S, min_V])
    maxHSV = np.array([max_H, max_S, max_V])
    segmented_temp = cv2.inRange(imgHSV, minHSV, maxHSV)
    segmented = cv2.bitwise_or(segmented, segmented_temp)
    cv2.imshow('resized_segmented', segmented)

    # HSV vizualizacio
    visualize_hsv_segment_parameters(pixel_HSV, vis_min_H, vis_max_H, min_S, max_S, min_V, max_V)


def segmentHSVRegion(x, y):
    global lastCoord, imgHSV, regionSelected, hsv_palette_bgr
    global diff_H, diff_S, diff_V, actualPixelHSV

    # Befoglalo teglalap bal felso, jobb also koordinataja
    minx = min(lastCoord[0], x)
    miny = min(lastCoord[1], y)
    maxx = max(lastCoord[0], x)
    maxy = max(lastCoord[1], y)

    # Ha nem valodi teglalap, akkor nem folytatjuk
    if minx != maxx and miny != maxy:
        # Kijelolt teruleten talalhato HSV min, max ertekek
        imCut = imgHSV[miny:maxy, minx:maxx]
        cut_minH = np.min(imCut[:, :, 0])
        cut_maxH = np.max(imCut[:, :, 0])
        cut_minS = np.min(imCut[:, :, 1])
        cut_maxS = np.max(imCut[:, :, 1])
        cut_minV = np.min(imCut[:, :, 2])
        cut_maxV = np.max(imCut[:, :, 2])
        # pixel szegmentalas parametereinek szamitasa
        actualPixelHSV = (int((int(cut_maxH) + int(cut_minH)) / 2), int((int(cut_maxS) + int(cut_minS)) / 2), int((int(cut_maxV) + int(cut_minV)) / 2))
        diff_H = (cut_maxH - cut_minH) >> 1
        diff_S = (cut_maxS - cut_minS) >> 1
        diff_V = (cut_maxV - cut_minV) >> 1
        segmentHSVPoint()

        # Kijeloles alaphelyzetbe
        lastCoord = []
        regionSelected = False


def mouse_click(event, x, y, flags, param):
    # Globalis valtozok atvetele
    global img, imgHSV, hsv_palette_circle, actualPixelHSV
    global lastCoord, regionSelected, mouseClicked
    global diff_H, diff_S, diff_V

    if event == cv2.EVENT_LBUTTONDOWN:
        lastCoord = (x, y)
        regionSelected = False
        mouseClicked = True

    if event == cv2.EVENT_MOUSEMOVE:
        if mouseClicked:
            regionSelected = True
            origImgOverlay = img.copy()
            cv2.rectangle(origImgOverlay, lastCoord, (x, y), (0, 0, 192), 3)
            cv2.imshow('resized_original', origImgOverlay)

    if event == cv2.EVENT_LBUTTONUP:
        mouseClicked = False

        if not regionSelected:
            # Voros szinu szalkereszt a kattintas helyere
            origImgOverlay = img.copy()
            origImgOverlay[y, :] = [0, 0, 192]
            origImgOverlay[:, x] = [0, 0, 192]
            cv2.imshow('resized_original', origImgOverlay)
            # Szegmentalas
            actualPixelHSV = imgHSV[y, x]
            segmentHSVPoint()

        if regionSelected:
            # A regio teglalap mar ki van rajzolva, itt nem kell
            # Szegmentalasi parameterek szamitasa a regio alapjan
            # lastCoord adja az atellenes csucspontot
            segmentHSVRegion(x, y)


# Foprogram

#png és jpg fájlokat olvashatunk be
filetypes = (('jpg files', '*.jpg'), ('png files', '*.png'))

link = askopenfilename(filetypes=filetypes)
img = cv2.imread(link, cv2.IMREAD_COLOR)

assert img is not None, 'Nincs kivalasztott kep!'

#Gauss szűrés
img = cv2.GaussianBlur(img, (5, 5), sigmaX=2.0, sigmaY=2.0)
#Konvertálás HSV színtérbe
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

hsv_palette_circle = np.ndarray((HSV_SIZE_Y + HSV_VALUE_SLIDER_HEIGHT, HSV_SIZE_X, 3), np.uint8)
hsv_palette_circle_mask = np.ndarray((HSV_SIZE_Y + HSV_VALUE_SLIDER_HEIGHT, HSV_SIZE_X, 3), np.uint8)

print('Computing HSV palette image...')
for j in range(0, HSV_SIZE_Y + HSV_VALUE_SLIDER_HEIGHT):
    for i in range(0, HSV_SIZE_X):
        dist = math.sqrt((j - HSV_CENTER_Y) ** 2 + (i - HSV_CENTER_X) ** 2)
        if dist >= HSV_SIZE_X / 2:
            hsv_palette_circle[j, i] = [0, 0, 255]
            hsv_palette_circle_mask[j, i] = [0, 0, 0]
        else:
            hsv_palette_circle_mask[j, i] = [0, 0, 255]
            hsv_palette_circle[j, i, 2] = HSV_DEFAULT_V_VALUE
            hsv_palette_circle[j, i, 1] = dist * HSV_FACTOR
            angle = math.atan2((HSV_SIZE_Y - j - HSV_CENTER_Y), (i - HSV_CENTER_X)) / 2
            if angle < 0:
                hsv_palette_circle[j, i, 0] = math.degrees(angle) + 180
            else:
                hsv_palette_circle[j, i, 0] = math.degrees(angle)

print('Computing done.')
print('Usable keys: h, H, s, S, v, V, q, w, b')
print('Click a pixel or select a region using left mouse button.')

update_HSV_palette(HSV_DEFAULT_V_VALUE)
cv2.imshow('palette', hsv_palette_bgr)
cv2.imshow('resized_original', img)
cv2.setMouseCallback('resized_original', mouse_click)

counterW = 0
counterB = 0

while(True):
    key = cv2.waitKey(0)

    if key == ord('q'):
        break

    if key == ord('h'):
        if diff_H > 1:
            diff_H = diff_H - 1
        segmentHSVPoint()

    if key == ord('H'):
        if diff_H < 179:
            diff_H = diff_H + 1
        segmentHSVPoint()

    if key == ord('s'):
        if diff_S > 5:
            diff_S = diff_S - 5
        segmentHSVPoint()

    if key == ord('S'):
        if diff_S < 250:
            diff_S = diff_S + 5
        segmentHSVPoint()

    if key == ord('v'):
        if diff_V > 5:
            diff_V = diff_V - 5
        segmentHSVPoint()

    if key == ord('V'):
        if diff_V < 250:
            diff_V = diff_V + 5
        segmentHSVPoint()

    if key == ord('b'):
        black_patches_img = segmented.copy()
        number_of_black_patches_pix = np.sum(black_patches_img == 255)

        if counterB == 0:
            counterB = 1

        print('black part saved')
        #cv2.imwrite('segmented_white.jpg', black_patches_img)

    if key == ord('w'):
        white_patches_img = segmented.copy()
        number_of_white_patches_pix = np.sum(white_patches_img == 255)

        if counterW == 0:
            counterW = 1

        print('white part saved')
        #cv2.imwrite('segmented_black.jpg', white_patches_img)

    if counterB + counterW == 2:

        print("White pixels:" + str(number_of_white_patches_pix))
        print("Black pixels: " + str(number_of_black_patches_pix))
        print("White rate: " + str(number_of_white_patches_pix/(number_of_black_patches_pix+number_of_white_patches_pix)))
        print("Black rate: " + str(number_of_black_patches_pix/(number_of_white_patches_pix+number_of_black_patches_pix)))

        counterB = 0
        counterW = 0


cv2.destroyAllWindows()