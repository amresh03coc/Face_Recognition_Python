import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video = cv2.VideoCapture(0)

ansh_image= face_recognition.load_image_file("image/ansh.jpg")
ansh_encoding = face_recognition.face_encodings(ansh_image)[0]

amresh_image= face_recognition.load_image_file("image/amresh.jpg")
amresh_encoding = face_recognition.face_encodings(amresh_image)[0]

kuber_image= face_recognition.load_image_file("image/kuber.jpg")
kuber_encoding = face_recognition.face_encodings(kuber_image)[0]


lakshay_image= face_recognition.load_image_file("image/laksh.jpg")
lakshay_encoding = face_recognition.face_encodings(lakshay_image)[0]


hardik_image= face_recognition.load_image_file("image/hardik.jpg")
hardik_encoding = face_recognition.face_encodings(hardik_image)[0]


arnav_image= face_recognition.load_image_file("image/arnav.jpg")
arnav_encoding = face_recognition.face_encodings(arnav_image)[0]

known_face_encoding = [ansh_encoding,amresh_encoding,kuber_encoding,lakshay_encoding,hardik_encoding,arnav_encoding]

known_faces_names =["Ansh","Amresh","Kuber","Lakshay","Hardik","Arnav"]

students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []
s=True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")


f=open(current_date+'.csv','w+',newline='')
lnwriter=csv.writer(f)

while True:
    _,frame= video.read()
    small_frame = cv2.resize(frame, (0,0), fx=0.25,fy=0.25)
    rgb_small_frame=small_frame[:,:,::1]
    if s:
        face_locations=face_recognition.face_locations(rgb_small_frame)
        face_encodings=face_recognition.face_encodings(rgb_small_frame , face_locations)
        face_names=[]
        for face_encoding in face_encodings:
            matches =face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance=face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name=known_faces_names[best_match_index]
            face_names.append(name)
            if name in known_faces_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time=now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time,current_date])    
    cv2.imshow("attendence system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

video.release()
cv2.destroyAllWindows()
f.close()


