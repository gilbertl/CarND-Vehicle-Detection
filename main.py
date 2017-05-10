import matplotlib.image as mpimg
import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from moviepy.editor import VideoFileClip


from sklearn.preprocessing import StandardScaler
import math
import glob
import random
import time
import cv2
import numpy as np
import cv2
from skimage.feature import hog
import itertools
from scipy.ndimage.measurements import label



def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        if not file.endswith('.png'):
            image = image.astype(np.float32)/255
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

CAR_IMAGES = [
    'training_data/vehicles/GTI_Far/*.png',
    'training_data/vehicles/GTI_Left/*.png',
    'training_data/vehicles/GTI_Right/*.png',
    'training_data/vehicles/GTI_MiddleClose/*.png',
    'training_data/vehicles/KITTI_extracted/*.png',
]

NON_CAR_IMAGES = [
    'training_data/non-vehicles/GTI/*.png',
    'training_data/non-vehicles/extras/*.png',
]

MAX_SAMPLE_SIZE = 99999999999999999999
#MAX_SAMPLE_SIZE = 500
COLORSPACE = 'YCrCb'
ORIENT = 18
PIX_PER_CELL = 8
CELL_PER_BLOCK = 3
HOG_CHANNELS = 'ALL'  # can be 0, 1, 2, or 'ALL'
SPATIAL_SIZE = (32, 32)
HIST_BINS = 32
TRAINING_CONFIG_PATH = 'training_config.p'



def train_car_non_car():
    cars = [path for pattern in CAR_IMAGES for path in glob.glob(pattern)]
    non_cars = [path for pattern in NON_CAR_IMAGES for path in glob.glob(pattern)]
    sample_size = min(len(cars), len(non_cars), MAX_SAMPLE_SIZE)
    # keep cars / non-cars sample size the same
    cars = random.sample(cars, sample_size)
    non_cars = random.sample(non_cars, sample_size)

    t = time.time()
    car_features = extract_features(cars, color_space=COLORSPACE, orient=ORIENT, 
                            pix_per_cell=PIX_PER_CELL, cell_per_block=CELL_PER_BLOCK, 
                            hog_channel=HOG_CHANNELS)
    non_car_features = extract_features(non_cars, color_space=COLORSPACE, orient=ORIENT, 
                            pix_per_cell=PIX_PER_CELL, cell_per_block=CELL_PER_BLOCK, 
                            hog_channel=HOG_CHANNELS)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract HOG features...')
    # Create an array stack of feature vectors
    X = np.vstack((car_features, non_car_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))


    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:',ORIENT,'orientations',PIX_PER_CELL,
        'pixels per cell and', CELL_PER_BLOCK,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

    training_config = {
        'svc': svc,
        'scaler': X_scaler,
        'orient': ORIENT,
        'pix_per_cell': PIX_PER_CELL,
        'cell_per_block': CELL_PER_BLOCK,
        'spatial_size': SPATIAL_SIZE,
        'hist_bins': HIST_BINS,
    }

    pickle.dump(training_config, open(TRAINING_CONFIG_PATH, 'wb'))

    return training_config
   

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    bboxes = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                bboxes.append([(xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart)])
                
    return bboxes

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def run_find_cars(training_config):
    svc = training_config["svc"]
    X_scaler = training_config["scaler"]
    orient = training_config["orient"]
    pix_per_cell = training_config["pix_per_cell"]
    cell_per_block = training_config["cell_per_block"]
    spatial_size = training_config["spatial_size"]
    hist_bins = training_config["hist_bins"]

    ystart = 400
    ystop = 656
    scale = 1.5

    class CarDetector():
        def __init__(self, img_range_255):
            self.bboxes_list = []
            self.img_range_255 = img_range_255
        
        def draw_vehicle_on_frame(self, img):
            if self.img_range_255:
                scaled_down_image = img.astype(np.float32) / 255

            self.bboxes_list.append(
                find_cars(scaled_down_image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))

            num_frames_avail = min(5, len(self.bboxes_list))

            # ouptut overlapping rectangles
            # output_img = np.copy(img)
            # for bboxes in self.bboxes_list[-num_frames_avail:]:
              # for bbox in bboxes:
                # cv2.rectangle(
                   # output_img, bbox[0], bbox[1], (0,0,255), 6)
            # return output_img

            heat = np.zeros_like(img[:,:,0]).astype(np.float)
            # Add heat to each box in box list
            
            for bboxes in self.bboxes_list[-num_frames_avail:]:
                heat = add_heat(heat, bboxes)

            # Apply threshold to help remove false positives
            heat = apply_threshold(heat, max(1, 1 * num_frames_avail * 0.5))
            # Visualize the heatmap when displaying    
            heatmap = np.clip(heat, 0, 255)
            # Find final boxes from heatmap using label function
            labels = label(heatmap)
            
            return draw_labeled_bboxes(np.copy(img), labels)


    img_paths = glob.glob('test_images/*.jpg')
    num_img_paths = len(img_paths)
    num_cols = 1
    num_rows = math.ceil(num_img_paths / num_cols)
    
    plt.figure(figsize=(30, 15))
    subplot_idx = num_img_paths * 100 + 20 + 1
    for i, img_path in enumerate(img_paths):
        img = mpimg.imread(img_path)
        car_detector = CarDetector(img_range_255=not img_path.endswith('.png'))
        out_img = car_detector.draw_vehicle_on_frame(img)

        plt.imsave("output_images/output_{}.jpg".format(i), out_img)
        plt.subplot(subplot_idx + i)
        plt.imshow(out_img)
    plt.show()

"""
    original_clip = VideoFileClip('project_video.mp4')
    car_detector = CarDetector(img_range_255=True)
    clip_with_overlays = original_clip.fl_image(car_detector.draw_vehicle_on_frame)
    clip_with_overlays.write_videofile('output_images/project_video.mp4', audio=False)
"""


def main():
    training_config = (train_car_non_car() 
        if len(sys.argv) > 1 and sys.argv[1] == '--retrain' 
        else pickle.load(open(TRAINING_CONFIG_PATH, 'rb')))
    run_find_cars(training_config)
    

if __name__ == '__main__':
    main()
