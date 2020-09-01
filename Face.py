from tkinter import *
import tkinter
from tkinter import filedialog 
from PIL import Image,ImageTk
import os
#import recognize_faces_image as rec
import threading
import cv2
import face_recognition
import argparse
import pickle
from tkinter.ttk import *
from tkinter import messagebox
from imutils import paths
import shutil 
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

def encode_faces(dataset,encodingsPath,detection_method):
    buttonBots["state"]=tkinter.DISABLED
    button_Detect_face["state"]=tkinter.DISABLED
    button_explore["state"]=tkinter.DISABLED
    button_train["state"]=tkinter.DISABLED
    button_Start["state"]=tkinter.DISABLED

    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images(dataset))

    # initialize the list of known encodings and known names
    knownEncodings = []
    knownNames = []
    threading.Thread(target=bar2,args=(len(imagePaths),)).start()
    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1,
            len(imagePaths)))
        global Counter1
        Counter1=i+1
        name = imagePath.split(os.path.sep)[-2]

        # load the input image and convert it from RGB (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb,
            model=detection_method)
        

        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)

        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and
            # encodings
            knownEncodings.append(encoding)
            knownNames.append(name)

    # dump the facial encodings + names to disk
    print("[INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    print(type(encodingsPath))
    f = open(encodingsPath, "wb")
    f.write(pickle.dumps(data))
    f.close()
    buttonBots["state"]=tkinter.NORMAL
    button_Detect_face["state"]=tkinter.NORMAL
    button_explore["state"]=tkinter.NORMAL
    button_train["state"]=tkinter.NORMAL

def browseFiles(): 
    load=Image.open('Example.PNG')
    render=ImageTk.PhotoImage(load)
    New_Win=Toplevel() 
    img=Label(New_Win,image=render)
    img.image=render
    img.grid(row=0,column=0)
    global filename
    filename= filedialog.askdirectory()
    imagePaths = list(paths.list_images(filename))
    label_file_explorer.configure(text="File Opened: "+filename) 
    New_Win.destroy()
    LabelCounter.configure(text="0"+ "/"+str(len(imagePaths)))
    button_train["state"]=tkinter.NORMAL

def pickleSave():
    global pickle
    pickle=filedialog.askdirectory()

def modelTrain():
    if (botFlag==True):
        New_Win=Toplevel()
        New_Win.geometry("200x150")
        LabelErr=Label(New_Win,text="First choose number of bots") 
        LabelErr.place(relx=0.5, rely=0.4, anchor=CENTER)
    else:
        #need to split to resources
        threading.Thread(target=encode_faces,args=(filename,'encoding2.pickle',"hog",)).start()
    #encode_faces(filename,'encoding2.pickle',"hog")

    
def bar2(NumPhotos): 
    while ((Counter1/NumPhotos)!=1):
        #percentage=Counter2/NumPhotos
        progress['value'] = int(100*(Counter1/NumPhotos))
        LabelCounter.configure(text=str(Counter1)+ "/"+str(NumPhotos))
    if (Counter1/NumPhotos)==1: 
        progress['value'] = int(100*(Counter1/NumPhotos))
        LabelCounter.configure(text="Complete " + str(Counter1)+ "/"+str(NumPhotos))


def bar(NumPhotos): 
    while ((Counter2/NumPhotos)!=1):
        #percentage=Counter2/NumPhotos
        progress['value'] = int(100*(Counter2/NumPhotos))
        LabelCounter.configure(text=str(Counter2)+ "/"+str(len(images_rgb)))
    if (Counter2/NumPhotos)==1: 
        progress['value'] = int(100*(Counter2/NumPhotos))
        LabelCounter.configure(text="Complete " + str(Counter2)+ "/"+str(len(images_rgb)))

  
    
# def Counter():        
#     while True:
#         label_file_explorer.configure(text=str(Counter)) 
#         if CounterFlag==True: break
    
    
def load_images_from_folder(folder):
    global images_rgb
    global images
    global filenames
    images_rgb = []
    images=[]
    filenames=[]
    for filename in os.listdir(folder):
        if (os.path.getsize(os.path.join(folder,filename))/1024)/1024>=3 : 
            continue
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            #print("NOT NONE")
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            filenames.append(os.path.join(folder,filename))
            images_rgb.append(rgb)
            images.append(img)
    LabelCounter.configure(text="0/"+str(len(images_rgb)))

    #print(filenames)
    #return images_rgb,images,filenames

def rec_Start(rec_encodings,detection_method,rgb_files,images,filenames):
    
    #print("[INFO] loading encodings...")
    data = pickle.loads(open(rec_encodings, "rb").read())
    # load the input image and convert it from BGR to RGB
    #print(data)
    names_dup= list(dict.fromkeys(data["names"]))
    #rgb_files,images,filenames=load_images_from_folder(folder)
    for j  in range (0,int(len(rgb_files))):
        #print(j+1)

        global Counter2
        Counter2+=1
        #label_file_explorer.configure(text=str(Counter)) 
        rgb=rgb_files[j]
    # detect the (x, y)-coordinates of the bounding boxes corresponding
    # to each face in the input image, then compute the facial embeddings
    # for each face
        #print("[INFO] recognizing faces...")
        boxes = face_recognition.face_locations(rgb,model=detection_method)
        encodings = face_recognition.face_encodings(rgb, boxes)

        # initialize the list of names for each face detected
        names = []

        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"],
                encoding,tolerance=0.5)
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number of
                # votes (note: in the event of an unlikely tie Python will
                # select first entry in the dictionary)
                name = max(counts, key=counts.get)

            # update the list of names
            names.append(name)
        #print(names)
        flag=False
        for train_name in names_dup:
            #print(train_name)
            if train_name not in names:
                flag=True
                break
                
        if flag==True:
            flag=False
            #filenames.remove(filenames[i])
            continue
        #print("Got here")
        #print(filenames[i])
        Detected_Images.append(filenames[j])
    finish.append(True)


def start():
     threading.Thread(target=realStart,args=()).start()
        
def realStart():
    buttonBots["state"]=tkinter.DISABLED
    button_Detect_face["state"]=tkinter.DISABLED
    button_explore["state"]=tkinter.DISABLED
    button_train["state"]=tkinter.DISABLED
    button_Start["state"]=tkinter.DISABLED
    #images_rgb,images,filenames=load_images_from_folder(filename2)
    bots=var.get()
    if bots>=len(images_rgb) : 
        bots=len(images_rgb)
        lenn=bots
    else:         
        lenn=int(len(images_rgb)/bots)

    count=0
    lenn2=int(bots)
    #print(lenn)
    threading.Thread(target=bar,args=(len(images_rgb),)).start()
    global finish
    finish=[]
    for i in range(0,lenn): 
        if (i==lenn and lenn2<len(images_rgb)): lenn2=len(images_rgb)
        x = threading.Thread(target=rec_Start, args=(('encoding2.pickle',"hog",images_rgb[count:lenn2],images[count:lenn2],filenames[count:lenn2],)))
        x.start()
        count=lenn2
        lenn2+=lenn     

    while len(finish)!=lenn:
        continue

    New_Win=Toplevel() 
    New_Win.geometry("600x500")
    newwinLabel=Label(New_Win,text="Click twice to see the Image")
    newwinLabel.place(relx=0.5, rely=0.1, anchor=CENTER)
    global Lb1
    Lb1 = Listbox(New_Win,selectmode=MULTIPLE)
    #print(len(Detected_Images))
    for i in range(0,len(Detected_Images)):
        Lb1.insert(i, Detected_Images[i])
    scrollbar = Scrollbar(New_Win) 
    scrollbar.pack(side = RIGHT, fill = BOTH) 
    Lb1.config(yscrollcommand = scrollbar.set) 
    scrollbar.config(command = Lb1.yview) 
    Lb1.pack()
    Lb1.bind("<Double-Button-1>", OnDouble)
    Lb1.place(relx=0.5, rely=0.4, anchor=CENTER)
    Save_button=Button(New_Win,text="Save chosen photos",command=savePhotos)
    Save_button.place(relx=0.5, rely=0.9, anchor=CENTER)

def savePhotos():
    saving=Lb1.curselection()
    SaveingPath= filedialog.askdirectory()
    for j in saving:
        shutil.copy(Detected_Images[j],SaveingPath)
    New_Win=Toplevel()
    New_Win.geometry('200x150')

    LabelErr=Label(New_Win,text="Done") 
    LabelErr.place(relx=0.5, rely=0.4, anchor=CENTER)

def OnDouble(event):
    widget = event.widget
    selection=widget.curselection()
    value = widget.get(selection[0])
    image = cv2.imread(value)
    cv2.imshow("Image",image)
    cv2.waitKey(0)

        
def sel():
    global botFlag
    botFlag=False
    selection = "Number of bots is = " + str(var.get())
    labelBots.config(text = selection)
    try: 
        if filename2!=None:
            button_Start["state"]=tkinter.NORMAL
    except: ""


def detect_Image():
    global filename2
    filename2= filedialog.askdirectory()
    threading.Thread(target=load_images_from_folder,args=(filename2,)).start()
    try: 
        if var.get()!=None: button_Start["state"]=tkinter.NORMAL
    except: ""

def openLinkedin():
    import webbrowser
    webbrowser.open("https://www.linkedin.com/in/yarden-molcho/")



root=Tk()
root.title("Face recognition - Yarden Molcho")
root.geometry('400x600')



global CounterFlag
CounterFlag=False
global Detected_Images
Detected_Images=[]
global Counter2
Counter2=0
global Counter1
Counter1=0
global botFlag
botFlag=True

variable = tkinter.DoubleVar(root)

progress = Progressbar(root, orient = HORIZONTAL, 
              length = 100, mode = 'determinate') 

photo = PhotoImage(file = 'Linkedin.PNG') 

var = DoubleVar()
scale = tkinter.Scale( root, variable = var,from_ = 1, to = 100,orient = HORIZONTAL) 
scale.place(relx=0.5, rely=0.7, anchor=CENTER)
label_file_explorer=Label(root,text="Hello " + os.getenv('username'))

button_explore = Button(root,text = "Browse Files",command = browseFiles)  
button_train = Button(root,text = "First train the model",command = modelTrain)  
button_Detect_face= Button(root,text = "Face Detection",command = detect_Image)  
buttonBots = Button(root, text="Confirm Number of bots", command=sel)
linkedinBtn=Button(root,text="My Linkedin",image=photo,command=openLinkedin)

button_Start = Button(root,text = "Start",command = start)  
button_explore.place(relx=0.5, rely=0.3, anchor=CENTER)
button_train.place(relx=0.5, rely=0.4, anchor=CENTER)
button_Start.place(relx=0.5, rely=0.6, anchor=CENTER)
button_Detect_face.place(relx=0.5, rely=0.5, anchor=CENTER)
progress.place(relx=0.5, rely=0.12, anchor=CENTER)
LabelCounter=Label(root,text="Progress bar : 0/0 ")
LabelCounter.place(relx=0.5, rely=0.05, anchor=CENTER)
linkedinBtn.place(relx=0.5, rely=0.93, anchor=CENTER)

label_file_explorer=Label(root,text="Hello " + os.getenv('username'))
label_file_explorer.place(relx=0.5, rely=0.2, anchor=CENTER)
labelBots=Label(root)
buttonBots.place(relx=0.5, rely=0.78, anchor=CENTER)
labelBots.place(relx=0.5, rely=0.83, anchor=CENTER)

#buttonBots["state"]=tkinter.DISABLED
#button_Detect_face["state"]=tkinter.DISABLED
#button_explore["state"]=tkinter.DISABLED
button_train["state"]=tkinter.DISABLED
button_Start["state"]=tkinter.DISABLED


root.mainloop()