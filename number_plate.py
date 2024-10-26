
import cv2 #computer vision model

harcascade = "F:\pwskills project\Car Number Plate Detection\model\haarcascade_russian_plate_number.xml"
cap = cv2.VideoCapture(0) # Capture webcam

cap.set(3,640) # set width
cap.set(4,480) # set height

min_area = 500 # min area for a detected region to be considered as a licence plate
count = 0

while True: # infinete loop
    #reading frames from webcam
    success, img = cap.read()

    #craeting a license plate classifier
    plate_cascade = cv2.CascadeClassifier(harcascade)

    #converting the frame to greyscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    # iterating through the detected plates
    for  (x, y, w, h) in plates:
        area = w*h
        # if  the area of the detected region is greater than the min area
        # draw a rectangle the plate and display the 
        # text "License Plate" on the top left corner of the rectangle.
        # Also display the region of intrest (ROI) of licence plate.
        if  area > min_area:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # display the text "License Plate" on the top left corner of the rectangle.
            cv2.putText(img, "License Plate", (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
            # display the region of intrest (ROI) of licence plate.
            img_roi = img[y: y+h, x:x+w]
            cv2.imshow("ROI", img_roi)
        cv2.imshow("Result", img)

        # Saving the plate when 's' is pressed
        # waitkey (1) means  that the program will wait for 1 millisecond
        # and then check if the  's' key is pressed. if it is pressed 
        # program  will save the plate and display the text "plate Saved"

        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite("F:\pwskills project\Car Number Plate Detection\plates\plate_"+str(count)+".jpg", img_roi)
            cv2.rectangle(img, (0, 200), (640, 300), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
            cv2.imshow("Result", img)
            cv2.waitKey(500)
            count += 1
        


