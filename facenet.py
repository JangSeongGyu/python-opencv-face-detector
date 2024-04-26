import cv2

def videoDetector(cam,cascade,age_net,gender_net,MODEL_MEAN_VALUES,age_list,gender_list):
    count = 1
    sum_age = 0
    cam.set(1,10)
    while True:
        
        ret,img = cam.read()
        try:
            img = cv2.resize(img,dsize=None,fx=1.0,fy=1.0)
        except: break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        results = cascade.detectMultiScale(gray,           
                                           scaleFactor= 1.1,
                                           minNeighbors=3, 
                                           minSize=(20,20)  
                                           )

        for box in results:
            x, y, w, h = box
            face = img[int(y):int(y+h),int(x):int(x+h)].copy()
            blob = cv2.dnn.blobFromImage(face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            # gender detection
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_preds.argmax()
            # Predict age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            
            age=0.0
            for index,data in enumerate(age_preds[0]):
                print(age_list[index],":",round(data,4)*100,"%")
                age+= age_list[index]*round(data,4)
            
            sum_age += age
            
            div_age = round(sum_age/count,0)
            info = gender_list[gender] + "age = " + str(div_age)
            count +=1
            print(age)
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), thickness=2)
            cv2.putText(img,info,(x,y-15),0, 0.5, (0, 255, 0), 1)


        cv2.imshow('facenet',img)

        if cv2.waitKey(1) > 0: 

            break

cascade_filename = 'haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascade_filename)


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

age_net = cv2.dnn.readNetFromCaffe(
	'deploy_age.prototxt',
	'age_net.caffemodel')

gender_net = cv2.dnn.readNetFromCaffe(
	'deploy_gender.prototxt',
	'gender_net.caffemodel')

age_list = [5,13,18,25,30,41,60,90]
gender_list = ['Male', 'Female']

cam = cv2.VideoCapture(0)

videoDetector(cam,cascade,age_net,gender_net,MODEL_MEAN_VALUES,age_list,gender_list )
