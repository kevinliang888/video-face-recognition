import cv2
import os
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import pickle
import numpy as np
import face_recognition
from create_dataset import get_path_list


#Get all the required path
dirpath = os.getcwd()
video_path = dirpath + "\\video"


def get_models():
    gender_model_path = dirpath + "/gender_model/gender.model"
    gender_lb_path = dirpath + "/gender_model/lb_gender.pickle"

    character_model_path = dirpath + "/character_model/character.model"
    character_lb_path = dirpath + "/character_model/lb_character.pickle"


    #import the model
    gender_model = load_model(gender_model_path)
    gender_lb = pickle.loads(open(gender_lb_path, "rb").read())

    character_model = load_model(character_model_path)
    character_lb = pickle.loads(open(character_lb_path, "rb").read())

    return character_model, character_lb, gender_model, gender_lb


def is_change_scene(frame1, frame2, threshold):

    if frame1 is None or frame2 is None:
        return False

    # extract a 3D RGB color histogram from the frame,
	# using 8 bins per channel, normalize, and update
	# the index
    hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8],
		[0, 256, 0, 256, 0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()


    hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8],
		[0, 256, 0, 256, 0, 256])
    hist2 = cv2.normalize(hist2, hist2).flatten()

    # cv2.HISTCMP_CORREL: Computes the correlation between the two histograms.
    # compute the distance between the two histograms
    d = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return d < threshold


def video_detect(video):
    shot_count = 1
    font = cv2.FONT_HERSHEY_DUPLEX
    video_name = video[:video.find(".")]

    # The output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_movie = cv2.VideoWriter(video_name + '_output.avi', fourcc, 30, (1280, 720))

    # Open the input movie file
    print(video_path + "\\" + video)
    input_movie = cv2.VideoCapture(video_path + "\\" + video)
    length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))


    # Initialize some variables
    frame_number = 0
    prev_frame = None
    frame = None

    # Import the character and gender model
    character_model, character_lb, gender_model, gender_lb = get_models()


    while True:

        # Grab a single frame of video
        prev_frame = frame
        ret, frame = input_movie.read()
        frame_number += 1

        #check if the scene changes
        if prev_frame is None or is_change_scene(prev_frame, frame, 0.95):
            if is_change_scene(prev_frame, frame, 0.95):
                shot_count += 1
        shot_label = "Shot {}".format(shot_count)
        cv2.putText(frame, shot_label, (10, 25), font, 1, (0, 255, 0), 2)
        #cv2.imwrite(shot_path +  "/shot" +str(shot_count)+".png",frame)


        # Quit when the input video file ends
        if not ret:
            break

        faces_locations = face_recognition.face_locations(frame)

        for (top, right, bottom, left) in faces_locations:

            # pre-process the image for classification
            crop_img = frame[top:bottom, left:right]
            crop_img = cv2.resize(crop_img, (96, 96))
            crop_img = crop_img.astype("float") / 255.0
            crop_img = img_to_array(crop_img)
            crop_img = np.expand_dims(crop_img, axis=0)

            # Use trained model to predict the character's name and gender
            proba = character_model.predict(crop_img)[0]
            idx = np.argmax(proba)
            character_label = character_lb.classes_[idx]

            proba = gender_model.predict(crop_img)[0]
            idx = np.argmax(proba)
            gender_label = gender_lb.classes_[idx]

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, character_label, (left + 2, bottom - 6), font, 0.8, (255, 255, 255), 1)
            cv2.putText(frame, gender_label, (left + 2, bottom + 20), font, 0.8, (255, 255, 255), 1)

        # Write the resulting image to the output video file
        output_movie.write(frame)
        print("Writing frame {} / {}".format(frame_number, length))

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
    video_detect(video)


if __name__ == '__main__':
    main()




