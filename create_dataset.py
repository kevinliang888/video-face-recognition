import face_recognition
import cv2
import os

# Get the path list for the documents in path
def get_path_list(path):
    path_list = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        path_list.extend(filenames)
        return path_list


def get_known_faces(face_path):
    names = []
    known_faces = []
    face_path_list = get_path_list(face_path)
    for img in face_path_list:
        name = img[:img.find(".")]
        picture = face_recognition.load_image_file(face_path + "\\" + img)
        picture_face_encoding = face_recognition.face_encodings(picture)[0]

        names.append(name)
        known_faces.append(picture_face_encoding)
    return names, known_faces


# Create the face dataset
def create_data(face_path, video_path):
    names, known_faces = get_known_faces(face_path)
    video_path_list = get_path_list(video_path)
    counters = [0 for i in range(len(names))]

    for video in video_path_list:
        video_name = video[:video.find(".")]

        # The output video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_movie = cv2.VideoWriter(video_name + '_output.avi',
                                       fourcc, 30, (1280, 720))
        # Open the input movie file
        print(video_path + "\\" + video)
        input_movie = cv2.VideoCapture(video_path + "\\" + video)
        length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize some variables
        face_locations = []
        face_encodings = []
        face_names = []
        frame_number = 0
        current_path = os.getcwd()

        while True:
            # Grab a single frame of video
            ret, frame = input_movie.read()
            frame_number += 1
            # Quit when the input video file ends
            if not ret:
                break
            # Find all the faces and face encodings in the
            # current frame of video
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame,
                                                             face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                match = face_recognition.compare_faces(known_faces,
                                                       face_encoding,
                                                       tolerance=0.50)
                # Find the name of the matching person
                name = None
                for i in range(len(match)):
                    if match[i]:
                        name = names[i]
                face_names.append(name)

            # Label the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                if not name:
                    continue

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                crop_img = frame[top:bottom, left:right]
                i = names.index(name)
                cv2.imwrite(current_path + "/character_dataset/" +
                            name + "/" + name +str(counters[i])+".png",crop_img)
                counters[i] += 1

                # Draw a label with a name below the face
                # cv2.rectangle(frame, (left, bottom - 35), (right, bottom),
                #               (0, 0, 255), cv2.FILLED)
                # font = cv2.FONT_HERSHEY_DUPLEX
                # cv2.putText(frame, name, (left + 6, bottom - 6), font,
                #             1.0, (255, 255, 255), 1)

            # Write the resulting image to the output video file
            output_movie.write(frame)
            print("Writing frame {} / {}".format(frame_number, length))

            cv2.imshow('face_recog', frame)
            # Hit 'q' on the keyboard to quit.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # All done!
        input_movie.release()
        cv2.destroyAllWindows()


def main():
    dirpath = os.getcwd()
    face_path = dirpath + "\\known_face"
    video_path = dirpath + "\\video"
    create_data(face_path, video_path)

if __name__ == '__main__':
    main()
