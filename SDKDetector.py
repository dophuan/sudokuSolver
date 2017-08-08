import cv2
import numpy as np
import argparse
import joblib as jb
import imutils
import random

######################################################################################################
#### Project 2 - Solving simple sodoku using image processing.
#### Introduction to computer vision
#### Course's ID: CS231.H21.KHTN
#### Lecturer: Dr. Ngo Duc Thanh and Msc Nguyen Thi Bao Ngoc
#### Authors: Do Phu An - 14520002, Hoang Minh Quan - 14520725, Luu Thanh Son - 14520772
######################################################################################################

#### This project includes 5 steps:
#### - Step 1: Detect the area that contains sudoku matrix
#### In this step, we use the max area algorithm. After converted the color input image into grayscale one,
#### we detect all contours in the grayscale image. The largest contour in square is the area that contains
#### sudoku matrix. (See in the code comment).

#### - Step 2: Warping sudoku area detected in step 1 into a flat-square image of sudoku matrix.

#### - Step 3: Extracting 81 blocks in sudoku matrix.
#### After finished step 2, we have a square image. Now we'll extract all number blocks in this image
#### for number digit detection in step 4.

#### - Step 4: Number digit detection.
#### In this step, we use Support Vector Machine (SVM) to detect number.
#### For details: Read the article: "Number digit detection using SVM"

#### - Step 5: Solving sudoku and displaying result.


# font to display on the result image.
font = cv2.FONT_HERSHEY_SIMPLEX

#Load number digit detection training file.
clf = jb.load('SVM3.pkl')

#=================================================================================================
#This code to run the program from cmd
ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required = True, help = "Path to query image")

args = vars(ap.parse_args())
#=================================================================================================


# Sudoku solver using backtracking algorithm
def findNextCellToFill(grid, i, j):
    for x in range(i, 9):
        for y in range(j, 9):
            if grid[x][y] == 0:
                return x, y
    for x in range(0, 9):
        for y in range(0, 9):
            if grid[x][y] == 0:
                return x, y
    return -1, -1


def isValid(grid, i, j, e):
    rowOk = all([e != grid[i][x] for x in range(9)])
    if rowOk:
        columnOk = all([e != grid[x][j] for x in range(9)])
        if columnOk:
            # finding the top left x,y co-ordinates of the section containing the i,j cell
            secTopX, secTopY = 3 * (i / 3), 3 * (j / 3)
            for x in range(secTopX, secTopX + 3):
                for y in range(secTopY, secTopY + 3):
                    if grid[x][y] == e:
                        return False
            return True
    return False


def solveSudoku(grid, i=0, j=0):
    i, j = findNextCellToFill(grid, i, j)
    if i == -1:
        return True
    for e in range(1, 10):
        if isValid(grid, i, j, e):
            grid[i][j] = e
            if solveSudoku(grid, i, j):
                return True
            # Undo the current cell for backtracking
            grid[i][j] = 0
    return False

# Image segmentation function
def segmentImage(img, thres):

    image = img

    row = image.shape[0]
    col = image.shape[1]

    J = row * col
    Size = row, col, 1
    R = np.zeros(Size, dtype=np.uint8)
    D = np.zeros((J, 3))
    arr = np.array((1, 3))

    counter = 0
    iter = 0.1

    threshold = 70
    currentMeanRandom = True
    currentMeanArr = np.zeros((1, 3))
    belowThresholdArr = []

    for i in range(0, row):
        for j in range(0, col):
            arr = image[i][j]

            for k in range(0, 3):
                if (k == 0):
                    D[counter][k] = image[i][j]
                else:
                    if (k == 1):
                        D[counter][k] = i
                    else:
                        D[counter][k] = j
            counter += 1

    while (len(D) > 0):
        #print len(D)

        if (currentMeanRandom):
            currentMean = random.randint(0, len(D) - 1)
            for i in range(0, 3):
                currentMeanArr[0][i] = D[currentMean][i]
        belowThresholdArr = []
        for i in range(0, len(D)):
            euclidDistance = 0

            for j in range(0, 3):
                euclidDistance += ((currentMeanArr[0][j] - D[i][j]) ** 2)
            euclidDistance = euclidDistance ** 0.5

            if (euclidDistance < threshold):
                belowThresholdArr.append(i)

        mean = 0
        mean_i = 0
        mean_j = 0
        currentMean = 0
        mean_col = 0

        for i in range(0, len(belowThresholdArr)):
            mean += D[belowThresholdArr[i]][0]
            mean_i += D[belowThresholdArr[i]][1]
            mean_j += D[belowThresholdArr[i]][2]

        mean = mean / len(belowThresholdArr)
        mean_i = mean_i / len(belowThresholdArr)
        mean_j = mean_j / len(belowThresholdArr)

        mean_e_distance = ((mean - currentMeanArr[0][0]) ** 2 + (mean_i - currentMeanArr[0][1]) ** 2 + (
            mean_j - currentMeanArr[0][2]) ** 2)

        mean_e_distance = mean_e_distance ** 0.5

        neareast_i = 0
        min_e_dist = 0
        counter_threshold = 0

        if (mean_e_distance < iter):
            newArr = np.zeros((1, 1))
            if mean < thres:
                #newArr[0][0] = mean
                newArr[0][0] = 255
            else:
                newArr[0][0] = 0

            for i in range(0, len(belowThresholdArr)):
                R[D[belowThresholdArr[i]][1]][D[belowThresholdArr[i]][2]] = newArr
                D[belowThresholdArr[i]][0] = -1
            currentMeanRandom = True
            newD = np.zeros((len(D), 3))
            counter_i = 0

            for i in range(0, len(D)):
                if (D[i][0] != -1):
                    newD[counter_i][0] = D[i][0]
                    newD[counter_i][1] = D[i][1]
                    newD[counter_i][2] = D[i][2]
                    counter_i += 1
            D = np.zeros((counter_i, 5))

            counter_i -= 1
            for i in range(0, counter_i):
                D[i][0] = newD[i][0]
                D[i][1] = newD[i][1]
                D[i][2] = newD[i][2]
        else:
            currentMeanRandom = False
            currentMeanArr[0][0] = mean
            currentMeanArr[0][1] = mean_i
            currentMeanArr[0][2] = mean_j
    return R


#========================================================================================================
# Main program
#========================================================================================================

# import an image given in the command line through the variable "query"
image = cv2.imread(args["query"])

# blur input image to eliminate all noise
image = cv2.GaussianBlur(image, (5,5), 0) # we can use meanshift algorithm at this line instead.

# convert input image into grayscale one
grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#detect the area that contains sudoku matrix
# in this step, we try to find the biggest contour area in the image

# create a mask for the image,
mask = np.zeros((grayImg.shape), np.uint8)

# find the kernel of this picture
# for details, try to google "cv2.getStructuringElement" and "cv2.MORPH_ELLIPSE"
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))

# as same as the line above
close = cv2.morphologyEx(grayImg,cv2.MORPH_CLOSE,kernel1)

# I don't exactly know what those code line below do >_<
# However, I think it looks like a filter
div = np.float32(grayImg)/(close)
res = np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))

res2 = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR)

# adaptive threshold function is a function that have the same effect as our image segmentation function.
# it means all 'light' points will be transformed into white points
# and 'dark' points will be transformed into black ones
threshold = cv2.adaptiveThreshold(res, 255, 0,1, 19,2)

# To show what do the adaptive threshold do, unblock 2 code lines below.
cv2.imwrite("Threshold.png", threshold)
#cv2.waitKey()


# get all bounding box in our image.
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#init
biggest = None
maxArea = 0
maxContour = None

# for each contour in collection of contours that we collected above,
# we calculate the square of them.
# If its square is larger than 1000, we compare it to the 'biggest' one.
for i in contours:
    area = cv2.contourArea(i)
    if area > 1000:
        #peri = cv2.arcLength(i, True)
        #approx = cv2.approxPolyDP(i, 0.02*peri, True)
        if area > maxArea: # and len(approx)==4:
            #biggest = approx
            maxArea = area
            maxContour = i

# Draw max contour border.
image2 = image.copy()
cv2.drawContours(image2, [maxContour], 0, (0, 191, 255), 3)
cv2.imwrite("MaxContourWithBorder.png", image2)

#after finding the biggest contours, now we have the sudoku area and the irrelevant ones
# we have to mix them by "AND" operator.
cv2.drawContours(mask, [maxContour], 0, 255 , -1)
cv2.imwrite("MaxContour.png", mask)
cv2.drawContours(mask, [maxContour], 0, 0, 2)
#cv2.imshow("Mask", mask)
res = cv2.bitwise_and(res, mask)

# to show what did it do, unblock 2 code lines below.
cv2.imwrite("SudokuMatrixAfterDetected.png", res)
#cv2.waitKey()


# Now, I will explain a little bit.
# Each sudoku matrix is created by 20 lines (10 vertical and 10 horizontal lines).
# and those lines will create 100 intersections.
# Therefore, we have to find all of them.

#finding vertical lines
kernelx = cv2.getStructuringElement(cv2.MORPH_RECT, (2,10))

dx = cv2.Sobel(res, cv2.CV_16S, 1, 0)
dx = cv2.convertScaleAbs(dx)

cv2.normalize(dx, dx, 0, 255, cv2.NORM_MINMAX)
ret, close = cv2.threshold(dx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
close = cv2.morphologyEx(close, cv2.MORPH_DILATE, kernelx, iterations = 1)

contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contour:
    x,y,w,h = cv2.boundingRect(cnt)
    if h/w > 5:
        cv2.drawContours(close,[cnt],0,255,-1)
    else:
        cv2.drawContours(close,[cnt],0,0,-1)
close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,None,iterations = 2)
closex = close.copy()

# find horizontal lines
kernely = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))
dy = cv2.Sobel(res,cv2.CV_16S,0,2)
dy = cv2.convertScaleAbs(dy)
cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
ret,close = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernely)

contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contour:
    x,y,w,h = cv2.boundingRect(cnt)
    if w/h > 5:
        cv2.drawContours(close,[cnt],0,255,-1)
    else:
        cv2.drawContours(close,[cnt],0,0,-1)

close = cv2.morphologyEx(close,cv2.MORPH_DILATE,None,iterations = 2)
closey = close.copy()

# Draw vertical contour
cv2.imwrite("VerticalLines.png", closex)

# Draw horizontal lines
cv2.imwrite("HorizontalLines.png", closey)

# after finding all vertical and horizontal lines, we will mix it by "AND" operator.
res = cv2.bitwise_and(closex, closey)

cv2.imwrite("VerticalAndHorizontalLinesMix.png", res)


img = image.copy()

# find the centroids. The collection of 'centroids' will be used to warp the input image to a flat-square image. ('square' in this case mean 'hinh vuong')

contour, hier = cv2.findContours(res,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
centroids = []
count = 0
for cnt in contour:
    mom = cv2.moments(cnt)
    (x,y) = int(mom['m10']/mom['m00']), int(mom['m01']/mom['m00'])
    cv2.circle(image,(x,y),4,(0,255,0),-1)
    centroids.append((x,y))
    cv2.putText(image, str(count), (x,y), font, 1, (255, 0, 0), 2)
    count += 1

cv2.imwrite("Centroids.png", image)

#sort the centroids. We have to sort all centroids because we may need to delete some of them
# in case we have a 'wrong' centroid.
centroids = np.array(centroids,dtype = np.float32)
c = centroids.reshape((count,2))
c2 = c[np.argsort(c[:,1])]

# solve distance between centroids
# if the distance between 2 centroids less than 20 px, one of them will be deleted.
# (Because we just need 100 'centroids'. Of course, the distance between them always greater than 20 px.

#init
prevX = 0
prevY = 0
temp = []
count = 1

for i in range(len(c2)):
    x, y = c2[i]
    if (((x-prevX)**2 + (y-prevY)**2)**0.5 > 20): # **2 = ^2, **0.5 = sqrt
        temp.append((x,y))
        cv2.circle(img, (x, y), 4, (255, 0, 0), -1)
        (prevX, prevY) = (x,y)
        cv2.putText(img, str(count), (x, y), font, 1, (255, 0, 0), 2)
        count += 1

cv2.imwrite("CentroidsAfterSorting.png", img)


c2 = np.array(temp,dtype = np.float32)

cv2.waitKey()

b = np.vstack([c2[i*10:(i+1)*10][np.argsort(c2[i*10:(i+1)*10,0])] for i in xrange(10)])
bm = b.reshape((10,10,2))

output = np.zeros((450,450,3),np.uint8)

# warp image, as same as warping image in panorama project.
for i,j in enumerate(b):
    ri = i/10
    ci = i%10
    if ci != 9 and ri!=9:
        src = bm[ri:ri+2, ci:ci+2 , :].reshape((4,2))
        dst = np.array( [ [ci*50,ri*50],[(ci+1)*50-1,ri*50],[ci*50,(ri+1)*50-1],[(ci+1)*50-1,(ri+1)*50-1] ], np.float32)
        retval = cv2.getPerspectiveTransform(src,dst)
        warp = cv2.warpPerspective(res2,retval,(450,450))
        output[ri*50:(ri+1)*50-1, ci*50:(ci+1)*50-1] = warp[ri*50:(ri+1)*50-1, ci*50:(ci+1)*50-1].copy()

cv2.imwrite("FlatSudoku.jpg", output)

# after warping the image, now we have a sudoku matrix as same as our eyes see.

input = []
count = 100
isZero = []

# now we extract all elements in this matrix.
# because every element has the same size ( width of image/9 and height of image / 9)
# it seems very easy :))))
for i in range(0,9):
    for j in range(0,9):

        # we calculate the location of top-right and bottom-left points of each element.
        x1 = j * (output.shape[0] / 9) + 5
        x2 = (j + 1) * (output.shape[0] / 9) - 5
        y1 = i * (output.shape[1] / 9) + 5
        y2 = (i + 1) * (output.shape[1] / 9) - 5

        # this function to 'cut' the element from the sudoku matrix
        element = output[y1:y2, x1:x2]
        #cv2.imwrite("Extracted\\" + str(count) + ".jpg", element)
        count += 1

        if (element.size != 0):

            element = cv2.cvtColor(element, cv2.COLOR_BGR2GRAY)
            # now we use our "cai lui" k-means algorithm haha.
            element = segmentImage(element, 220)
            element = cv2.resize(element, (36, 36))
            #element = cv2.cvtColor(element, cv2.COLOR_BGR2GRAY)


            # unfold the folder named "Extracted" to see what they do.
            # As I've told already, our 'k-mean' function looks like the adaptive threshold function.
            # to know more about image segmentation's application, try to google "semantic object detection"

            cv2.imwrite("Extracted\\" + str(count) + ".jpg", element)
            #element = cv2.adaptiveThreshold(element, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 101, 1)


            # use SVM to detect number.
            y = clf.predict(np.reshape(element, (1, -1)))
            input.append(y)
            if (y[0]!= 0):
                isZero.append(0)
                # if it is not equal to '0'. Display this number.
                cv2.putText(output, str(y[0]), ((x1 + x2) / 2, (y1 + y2) / 2), font, 1, (0, 0, 255), 2)
            else:
                #else, we import it to a list.
                isZero.append(1)

# resize the input matrix into another matrix that my sudoku solver can solve.
input = np.reshape(input, (9, -1))
solveSudoku(input)

# resize it one more time for displaying result.
input = np.reshape(input, (-1, 1))
isZero = np.reshape(isZero, (-1, 1))

#print result.
count = 0
for i in range(0,9):
    for j in range(0,9):
        x1 = j * (output.shape[0] / 9) + 5
        x2 = (j + 1) * (output.shape[0] / 9) - 5
        y1 = i * (output.shape[1] / 9) + 5
        y2 = (i + 1) * (output.shape[1] / 9) - 5
        temp = input[count]

        if isZero[count] == 1:
            cv2.putText(output, str(temp), (x1 - 5, (y1 + y2)/2 + 7), font, 1, (255,0,0), 2)
        count += 1

cv2.imshow("result", output)
cv2.waitKey()