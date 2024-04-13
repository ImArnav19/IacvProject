import cv2
import time
import datetime

#video capture devices , 0 for front Cam
cap = cv2.VideoCapture(0)

#cascade ml models by cv for faces and bodies
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_fullbody.xml")

detection = False       #detection boolean
detection_stopped_time = None  #initiated to avoid null errors
timer_started = False   
SECONDS_TO_RECORD_AFTER_DETECTION = 5   #after detection still keep recording

frame_size = (int(cap.get(3)), int(cap.get(4)))  #the size i want the video must be, its frame size cap.get(3)-width, cap.get(4)-Height

fourcc = cv2.VideoWriter_fourcc(*"mp4v")  #saving files mp4

#run loop continuosly
while True:

    #Read 30fps
    _, frame = cap.read()   #used _ does like we dont care what cap.read() gives, we just want frame

    #grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)             #gray=image,scaleFactor(1.3)=accuracy speed of algo,min no of neighbours(5) lower the number more faces it will catch
    bodies = body_cascade.detectMultiScale(gray, 1.3, 5)

    # get FPS of input video 
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # print(fps)
    # define output video and it's FPS 
    
    output_fps = 20

    if len(faces) + len(bodies) > 0:  #logic for sensing detection
        if detection:        #faces detected and recording
            timer_started = False
        else:                 #faces detected not recording
            detection = True
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")  #naming convention
            out = cv2.VideoWriter(
                f"{current_time}.mp4", fourcc, output_fps, frame_size)  #saving video recording
            print("Started Recording!")
    elif detection:            #no face detection but detected faces before
        if timer_started:
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                #stop everything as current time exceeded 5s with no faces
                detection = False
                timer_started = False
                out.release()
                print('Stop Recording!')
        else:
            #setup code to intiate variables to stop recording
            timer_started = True
            detection_stopped_time = time.time()

    if detection:
        out.write(frame)   #writing videos

    for (x, y, width, height) in faces:
       cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 0), 3)

    cv2.imshow("Camera", frame)   #shows the camera

    if cv2.waitKey(1) == ord('q'):     #stop program when pressed q
        break

#release resources
out.release()   #saves video after ending the script
cap.release()
cv2.destroyAllWindows()