import face_recognition
from create_dataset import get_path_list
from video_detect import get_models, is_change_scene
import cv2
import os
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import copy
from collections import OrderedDict
from scipy.spatial import distance as dist


#Get all the required path
dirpath = os.getcwd()
video_path = dirpath + "\\video"

class CenterTracker():
    def __init__(self, maxDisappeared=50):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared
        self.total_detect = 0
    def get_objects(self):
        return self.objects

    def reset(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.total_detect = 0

    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
        self.total_detect += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object
            # centroids and input centroids, respectively 
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # (1) find the smallest value in each row and then 
            # (2) sort the row indexes based on their minimum values
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # If the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        # return the set of trackable objects
        return self.objects


# Show how the head moves and how much it moves
def show_move(prev_centroid, centroid, name, cm_per_pixel):
    [x, y] = list(centroid)
    [x_prev, y_prev] = list(prev_centroid)
    move_string = ""
    if [x_prev, y_prev] == [x, y]:
        move_string = name + "'s head doesn't move."
    else:
        if x > x_prev:
            direction_x = "right"
        elif x < x_prev:
            direction_x = "left"
        else:
            direction_x = ""

        if y > y_prev:
            direction_y = "down"
        elif y < y_prev:
            direction_y = "up"
        else:
            direction_y = ""

        length_x = abs((x - x_prev) * cm_per_pixel)
        length_y = abs((y - y_prev) * cm_per_pixel)

        if direction_x == "" or direction_y == "":
            if direction_x == "":
                move_string = "{}'s head move {} {:.2f}cm".format(name,
                  direction_y, length_y)
            else:
                move_string = "{}'s head move {} {:.2f}cm".format(name,
                  direction_x, length_x)
        else:

            distance = np.sqrt(length_x ** 2 + length_y ** 2)
            move_string = "{}'s head move {} {} {:.2f}cm".format(name,
                  direction_y, direction_x, distance)
    return move_string


# Get all the rectangles and corresponding centers
def get_centers(face_locations, frame):
    x = []
    centers = []
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        (startX, startY, endX, endY) = (right, top, left, bottom)
        cX = int((startX + endX) / 2.0)
        cY = int((startY + endY) / 2.0)
        x.append((startX, startY, endX, endY))
        centers.append([cX, cY])
    return x, centers


# Transform the crop image to the type 
# that can be feed into the Neural network
def transform_crop_img(crop_img):
    crop_img = cv2.resize(crop_img, (96, 96))
    crop_img = crop_img.astype("float") / 255.0
    crop_img = img_to_array(crop_img)
    crop_img = np.expand_dims(crop_img, axis=0)
    return crop_img


# Detect and Track all the faces in the video
def video_track(video):
    shot_count = 1
    font = cv2.FONT_HERSHEY_DUPLEX
    video_name = video[:video.find(".")]

    # The output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_movie = cv2.VideoWriter(video_name + '_output.avi', fourcc, 30, (1280, 720))

    # Open the input movie file
    input_movie = cv2.VideoCapture(video_path + "\\" + video)

    # Initialize some variables
    frame_number = 0
    prev_frame = None
    frame = None
    # Store the track object
    object_name = {}
    objects = None

    #Initialize the tracker
    ct = CenterTracker()

    # Import the character and gender model
    character_model, character_lb, gender_model, gender_lb = get_models()


    while True:

        # Grab a single frame of video
        prev_frame = frame
        ret, frame = input_movie.read()
        #frame_copy = frame.copy()
        frame_number += 1
        print("Frame " + str(frame_number))


        #check if the scene changes
        if prev_frame is None or is_change_scene(prev_frame, frame, 0.95):
            if is_change_scene(prev_frame, frame, 0.95):
                object_name = {}
                ct.reset()
                objects = None
                shot_count += 1
        shot_label = "Shot {}".format(shot_count)
        cv2.putText(frame, shot_label, (10, 25), font, 1, (0, 255, 0), 2)
        #cv2.imwrite(shot_path +  "/shot" +str(shot_count)+".png",frame)

        # Quit when the input video file ends
        if not ret:
            break
        
            
        # Find all the detection and calculate centriod
        face_locations = face_recognition.face_locations(frame)
        x, centers = get_centers(face_locations, frame)

        prev_object = copy.deepcopy(objects)
        objects = ct.update(x)

        counter = 1
        for (objectID, centroid) in objects.items():
            # Check if we lose detection for the person
            if list(centroid) in centers:
                index = centers.index(list(centroid))
                (top, right, bottom, left) = face_locations[index]
                crop_img = frame[top:bottom, left:right]
                crop_img = transform_crop_img(crop_img)

                # Check if it is the first time to detect this person
                if objectID not in object_name:
                    # Predict the character name
                    proba = character_model.predict(crop_img)[0]
                    idx = np.argmax(proba)
                    character_label = character_lb.classes_[idx]
                    
                    #Predict the character gender
                    proba = gender_model.predict(crop_img)[0]
                    idx = np.argmax(proba)
                    gender_label = gender_lb.classes_[idx]
                    object_name[objectID] = [character_label, gender_label]

                [name, gender] = object_name[objectID]
                cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 2, bottom - 6), font, 0.8, (255, 255, 255), 1)
                cv2.putText(frame, gender, (left + 2, bottom + 20), font, 0.8, (255, 255, 255), 1)

                # Calculate relation between real world cm and pixel
                face_width = 11
                pixel_width = right - left
                # width per pixel
                cm_per_pixel = face_width / pixel_width

                if prev_object == None or objectID not in prev_object:
                    move_string = "{}'s head doesn't move".format(name)
                else:
                    prev_centroid = prev_object[objectID]
                    move_string = show_move(prev_centroid, centroid, name, cm_per_pixel)
            else:
                move_string =  "Lose detection for {}".format(object_name[objectID][0])

            cv2.putText(frame, move_string, (10, 25 + 40 * counter), font, 1, (255, 255, 0), 2)
            counter += 1
            print(move_string)

        print()
        # Write the resulting image to the output video file
        output_movie.write(frame)
        # Output the track image
        cv2.imwrite(os.getcwd() + "/track_img/" + "frame" +str(frame_number)+".png",frame)

        cv2.imshow('face_recog_crop', frame)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # All done!
    input_movie.release()
    cv2.destroyAllWindows()

def main():
    dirpath = os.getcwd()
    video_path = dirpath + "\\video"
    video_path_list = get_path_list(video_path)
    video = video_path_list[3]
    video_track(video)


if __name__ == '__main__':
    main()

