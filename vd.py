#
# Define a function that returns HOG features for an image
#
debug = True
import cv2
def GetHOGFeatures(img, feature_vector):
    global debug
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #if debug: print('gray type, shape', type(gray), gray.shape)
    # block_norm='L2',
    features = hog(gray, orientations=orient,\
                          pixels_per_cell=(pix_per_cell, pix_per_cell),\
                          cells_per_block=(cell_per_block, cell_per_block),\
                          transform_sqrt=True,\
                          feature_vector=False)
    #if debug: print('hog features type, shape', type(features), features.shape)

    return features
import cv2
import glob
#
# There is a temptation to use the different sets of car images for
# different windows in the image
#

fCarFiles = glob.glob('vehicles/GTI_far/*.png')
print ('Far car image filenames',len(fCarFiles), fCarFiles[0])

mCCarFiles = glob.glob('vehicles/GTI_MiddleClose/*.png')
print ('Middle close car image filenames',len(mCCarFiles), mCCarFiles[0])

rCarFiles = glob.glob('vehicles/GTI_right/*.png')
print ('Right car image filenames',len(rCarFiles), rCarFiles[0])

lCarFiles = glob.glob('vehicles/GTI_left/*.png')
print ('Left car image filenames',len(lCarFiles), lCarFiles[0])

kCarFiles = glob.glob('vehicles/KITTI_extracted/*.png')
print ('KITTI car image filenames',len(kCarFiles), kCarFiles[0])

nonCarFiles = glob.glob('non-vehicles/GTI/*.png')
print ('GTI non-Car image filenames',len(nonCarFiles), nonCarFiles[0])

nonECarFiles = glob.glob('non-vehicles/GTI/*.png')
print ('Extra non-car image filenames',len(nonECarFiles), nonECarFiles[0])

carFiles = fCarFiles + mCCarFiles + rCarFiles + lCarFiles + kCarFiles
print ('Car image filenames',len(carFiles), carFiles[0])

nonCarFiles = nonCarFiles + nonECarFiles
print ('Total non-Car image filenames',len(nonCarFiles), nonCarFiles[0])

from sklearn.utils import shuffle

carFiles = shuffle(carFiles)
nonCarFiles = shuffle(nonCarFiles)

numImages2Use = min(len(carFiles), len(nonCarFiles))

X = carFiles+nonCarFiles
y = [True]*len(carFiles)+[False]*len(nonCarFiles)

from skimage.feature import hog
pix_per_cell = 8
cell_per_block = 2
orient = 9


# From lessons
import numpy as np
# Define a function to compute color histogram features  
def color_hist(img, nbins=16, bins_range=(0, 256)):
    # Compute the histogram of the HSV channels separately
    hhist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    shist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    vhist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = hhist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((hhist[0], shist[0], vhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    #return hhist, shist, vhist, bin_centers, hist_features
    return hist_features

MAX_ITER = -1
from sklearn.preprocessing import StandardScaler
#
# For each image in the set, create features for color hist
#
print ('Calculating features for color')
Fcolor = []
for x in X:
    img = cv2.imread(x)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_features = color_hist(img)
    Fcolor += [hist_features]
#STACK = np.vstack(Fcolor).astype(np.float64)
# Fit a per-column scaler
#Fcolor_scaler = StandardScaler().fit(STACK)
# Apply the scaler to STACK
#Fcolor = Fcolor_scaler.transform(Fcolor)
print('Using', numImages2Use, 'images')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, F_train, F_test = train_test_split(\
    X, y, Fcolor, test_size=0.20)
print('Lengths', len(X_train), len(X_test), len(y_train), len(y_test),\
     len(F_train), len(F_test))
from sklearn.svm import SVC
print('Train the Color Support Vector Machine')
clfColor = SVC(max_iter=MAX_ITER)
clfColor.fit(F_train, y_train)
print('Test the Color Support Vector Machine')
pred = clfColor.predict(F_test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, y_test)
print('Color Accuracy = ', acc)
#
# For each image in the set, create features
#
print ('Calculating HOG features')
F = []
for x in X:
    img = cv2.imread(x)
    features = GetHOGFeatures(img, True).ravel()
    F += [features]
print(len(F), 'HOG features computed')
   
print('Using', numImages2Use, 'images')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, F_train, F_test = train_test_split(\
    X, y, F, test_size=0.20)
from sklearn.svm import SVC
print('Train the HOG Support Vector Machine')
clf = SVC(max_iter=MAX_ITER)
print('len F_train, y_train', len(F_train), len(y_train))
clf.fit(F_train, y_train)
print('Test the Support Vector Machine')
pred = clf.predict(F_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, y_test)

print('HOG Accuracy = ', acc)
#
# Define and train a full model
#
#
# Construct a combined feature vector
#

FcolorHOG = []
i = 0
for f1 in Fcolor:
    f2 = F[i]
    feature_list = [f1, f2]
    
    # Create an array stack, NOTE: StandardScaler() expects np.float64
    #STACK = np.vstack(feature_list).astype(np.float64)
    # Fit a per-column scaler
    #STACK_scaler = StandardScaler().fit(STACK)
    # Apply the scaler to STACK
    #scaled_STACK = STACK_scaler.transform(STACK)
    #print('f1',f1)
    #print('f2',f2)
    scaled_STACK = np.concatenate((f1, f2)) # Not NORMALIZED!!!!!
    FcolorHOG += [scaled_STACK]
    i += 1
# Create an array stack, NOTE: StandardScaler() expects np.float64
STACK = np.vstack(FcolorHOG).astype(np.float64)
# Fit a per-column scaler
STACK_scaler = StandardScaler().fit(STACK)
# Apply the scaler to STACK
FcolorHOG = STACK_scaler.transform(STACK)
X_train, X_test, y_train, y_test, F_train, F_test = train_test_split(\
    X, y, FcolorHOG, test_size=0.20)
print('Train the Color and HOG Support Vector Machine')
clfColorHOG = SVC(max_iter=MAX_ITER)
clfColorHOG.fit(F_train, y_train)
print('Test the Color Support Vector Machine')
pred = clfColorHOG.predict(F_test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, y_test)
print('ColorHOG Accuracy = ', acc)
#
# Try to find optimum parameters
#
'''
from sklearn import svm, grid_search, datasets
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 5, 10]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(F_train, y_train)

print('Best paramters', clf.best_params_)
print('Test the Best Found Support Vector Machine')
pred = clf.predict(F_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, y_test)

print('Best Found Accuracy = ', acc)
'''
cv2.waitKey(0)
cv2.destroyAllWindows()
#
# Save the model
#
from sklearn.externals import joblib
joblib.dump(clfColorHOG, 'ColorHOG.pkl') 
#
# Load the model
#
from sklearn.externals import joblib
clfCar = joblib.load('ColorHOG.pkl')
#
# Take some lesson code and modify it to work using my code
#
# Define a single function that can extract features using hog sub-sampling and make predictions
import cv2
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block):
    
    #print('find input image type, shape', type(img),img.shape)
    draw_img = np.copy(img)
    img_tosearch = img[ystart:ystop,:,:]
    #print('find img_tosearch type, shape', type(img_tosearch),img_tosearch.shape)
    #print('ystart, ystop', ystart, ystop)
    gray = img_tosearch #cv2.cvtColor(img_tosearch,cv2.COLOR_BGR2GRAY)
    #print('find gray type, shape', type(gray),gray.shape)
    hsv  = cv2.cvtColor(img_tosearch, cv2.COLOR_BGR2HSV)

    
    if scale != 1:
        imshape = gray.shape
        gray = cv2.resize(gray, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        hsv  = cv2.resize(hsv,  (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    
    # Define blocks and steps as above
    nxblocks = (gray.shape[1] // pix_per_cell)-1
    nyblocks = (gray.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image

    hog1 = GetHOGFeatures(gray, False)
    #print('hog1 type and shape', type(hog1), hog1.shape)

   
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_features = hog1[ypos:ypos+nblocks_per_window,\
                                xpos:xpos+nblocks_per_window]
            #print('find hog features type and shape', type(hog_features),\
            #      hog_features.shape)
            hog_features = hog_features.ravel() 
            #print('hog_features type', type(hog_features), hog_features.shape)
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(hsv[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            hist_features = np.array(color_hist(subimg))

            # Scale features and make a prediction

            test_features = X_scaler.transform(np.hstack((hist_features, hog_features)).reshape(1, -1)) 
            test_prediction = svc.predict(test_features)
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                
    return draw_img
#
# Test the find_cars function
#
ystart = 400
ystop  = 656
scale = 1.0

img = cv2.imread('test_images/test6.jpg')
out_img = find_cars(img, ystart, ystop, scale, clfCar, STACK_scaler, orient, pix_per_cell, cell_per_block)
cv2.imshow('Find Cars6', out_img)
cv2.waitKey(0)
img = cv2.imread('test_images/test5.jpg')
out_img = find_cars(img, ystart, ystop, scale, clfCar, STACK_scaler, orient, pix_per_cell, cell_per_block)
cv2.imshow('Find Cars5', out_img)
cv2.waitKey(0)
img = cv2.imread('test_images/test4.jpg')
out_img = find_cars(img, ystart, ystop, scale, clfCar, STACK_scaler, orient, pix_per_cell, cell_per_block)
cv2.imshow('Find Cars4', out_img)
cv2.waitKey(0)
img = cv2.imread('test_images/test3.jpg')
out_img = find_cars(img, ystart, ystop, scale, clfCar, STACK_scaler, orient, pix_per_cell, cell_per_block)
cv2.imshow('Find Cars3', out_img)
cv2.waitKey(0)
img = cv2.imread('test_images/test2.jpg')
out_img = find_cars(img, ystart, ystop, scale, clfCar, STACK_scaler, orient, pix_per_cell, cell_per_block)
cv2.imshow('Find Cars2', out_img)
cv2.waitKey(0)



img = cv2.imread('test_images/test1.jpg')
out_img = find_cars(img, ystart, ystop, scale, clfCar, STACK_scaler, orient, pix_per_cell, cell_per_block)

cv2.imshow('Find Cars', out_img)
cv2.imwrite('carsfound.jpg', out_img)
cv2.waitKey(0)
cv2.destroyAllWindows()