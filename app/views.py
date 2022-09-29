from ast import Pass
from crypt import methods
from unittest import result
from app import app 
from flask import render_template , request , session, redirect, url_for
import pymysql
from app.admin_views import admin_dashboard
import os 
from werkzeug.utils import secure_filename
import time

from flask import Flask, Response
import cv2
import datetime, time
import sys
import numpy as np
from threading import Thread
# import app.camera 
# import app.compare as cmp




from os import path
import sys
print(sys.version)
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())

accelerator = 'cu80' if path.exists('/opt/bin/nvidia-smi') else 'cpu'

import torch
print(torch.__version__)
print(torch.cpu)

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

connection = pymysql.connect(host = 'localhost', user = 'root', password = '', database = 'guardians')
cursor = connection.cursor()
app.secret_key = 'mysecretkey'

isCapture = 0

global username

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

class Config():
    training_dir = "/Users/alexandroalvin/Downloads/Training/"
    testing_dir = "/Users/alexandroalvin/Downloads/Testing"
    train_batch_size = 64
    train_number_epochs = 150

def find_classes(dir):
    classes = os.listdir(dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


class SiameseNetworkDataset(Dataset):
    
    def __init__(self,imageFolderDataset,transform=None,should_invert=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        
    def __getitem__(self,index):
        img0_tuple = random.choice([item for item in self.imageFolderDataset.imgs if item[1] < 2]) #Considering only genuine images for perfect pair
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1)
        if should_get_same_class:
          while True:
            img1_tuple = random.choice(self.imageFolderDataset.imgs)
            if img0_tuple[1]==img1_tuple[1]:
              break

        else:
          img1_tuple = random.choice([item for item in self.imageFolderDataset.imgs if item[1] == (img0_tuple[1] + 2)]) # Considering a pair of fake + genuine of same person's signature

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)

folder_dataset = dset.ImageFolder(root=Config.training_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*100*100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

train_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=8, #ss
                        batch_size=Config.train_batch_size)



net = SiameseNetwork().cpu()
# net = None
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.001)

counter = []
loss_history = [] 
iteration_number= 0


# filepath = "/Users/alexandroalvin/Documents/AI Project_Guardian Python/Datasets/Ssave1.pt"
filepath = "/Users/alexandroalvin/Documents/Guardians_FrontEnd/Datasets/epoch100.pt"


net.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))


in1 = None
in2 = None
class InferenceSiameseNetworkDataset(Dataset):
    def __init__(self,imageFolderDataset,transform=None,should_invert=True):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.should_invert = should_invert
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            while True:
                #keep looping till a different class image is found
                
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                # print (img1_tuple[0])
                if img0_tuple[1] !=img1_tuple[1]:
                    break
        img0 = Image.open(in1)
        img1 = Image.open(in2)
        # img0 = Image.open('/Users/alexandroalvin/Documents/Guardians_FrontEnd/Datasets/Group 1.png')
        # img1 = Image.open('/Users/alexandroalvin/Documents/Guardians_FrontEnd/Datasets/Rectangle 2.png')
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        # img0 = img0.resize((500,500),Image.ANTIALIAS)
        # img1 = img1.resize((500,500),Image.ANTIALIAS)

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)

def compares(usernames, directories):

    #Algoritma rata-rata
    compareName = usernames

    compareDir = cursor.execute("SELECT pathfile FROM karyawan WHERE namaKaryawan = %s", (compareName))
    compareDir = cursor.fetchone()
    print(compareDir)

    compareUploads = directories

    print(compareName)

    values = 0.0
    dataTotal = 0


    for i in range (5):
        global in1
        in1 = compareDir[0] + '/' + 'File_000.png'
        if (i > 0):
            in1 = compareDir[0] + '/' + 'File_000('+ str(i) + ').png'

        global in2
        in2 = compareUploads

        folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
        siamese_dataset = InferenceSiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                                transform=transforms.Compose([transforms.Resize((100,100)),
                                                                            transforms.ToTensor()
                                                                            ])
                                            ,should_invert=False)

        test_dataloader = DataLoader(siamese_dataset,num_workers=6,batch_size=1,shuffle=True)
        dataiter = iter(test_dataloader)
        x0,_,_ = next(dataiter)

        for i in range(1):
            _,x1,label2 = next(dataiter)
            concatenated = torch.cat((x0,x1),0)
            
            output1,output2 = net(Variable(x0).cpu(),Variable(x1).cpu())

            euclidean_distance = F.pairwise_distance(output1, output2)

            dissimilar = euclidean_distance.item()
            # imshow(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))
            print (dissimilar)
            
            values += dissimilar
            dataTotal += 1


    rate = float(values/dataTotal)
    print ("rate" +str(rate))

    if (rate < 0.85):
        print ("Match!")
        return True;
    else:
        print ("Not Match / Dissimilar!")
        return False;


app.config['IMAGE_UPLOADS'] = '/Users/alexandroalvin/Documents/Guardians_FrontEnd/app/static/img/uploads'
app.config['ALLOWED_IMAGE_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'gif'])


# global capture,rec_frame, grey, switch, neg, face, rec, out 
capture=0
cameraOn = 0
successCapture = 0



#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

camera = None
def runcam():
    global camera
    camera = cv2.VideoCapture(0)



def gen_frames():  # generate frame by frame from camera
    global out, capture, successCapture
    while True:
        success, frame = camera.read() 
        if success:   
            if(capture == 1):
                capture=0
                basedir = os.path.abspath(os.path.dirname(__file__))
                p = os.path.join(basedir, app.config["IMAGE_UPLOADS"], 'cameraCompare.png')
                cv2.imwrite(p, frame)
                print("Captured saved!")
                successCapture = 1
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
        else:
            pass


def stopcamera():
    global camera
    camera.release()
    cv2.destroyAllWindows()

@app.route('/fail',methods=['GET'])
def fail():
    if "username" in session:
        username = session['username']
        return render_template('public/fail.html', usernames = username)
    else:
        return redirect(url_for('login'))

@app.route('/success',methods=['GET'])
def success():
    if "username" in session:
        username = session['username']
        return render_template('public/success.html', usernames = username)
    else:
        return redirect(url_for('login'))

def allowed_image(filename):
    if not '.' in filename:
        return False

    ext = filename.rsplit('.', 1)[1]

    if ext.upper() in app.config['ALLOWED_IMAGE_EXTENSIONS']:
        return True
    else:
        return False

@app.route('/open-cam', methods=['GET', 'POST'])
def start_camera():
    if "username" in session:
        username = session['username']
        global cameraOn
        cameraOn = 1
        if cameraOn == 1:
            runcam()
        return render_template('public/camera.html', usernames = username)
    else:
        return redirect(url_for('login'))
    # runcam()
    # global cameraOn
    # cameraOn = 1
    # return render_template('public/camera.html')

@app.route('/upload-image',methods=['GET','POST'])
def upload_image():
    username = session['username']
    global isCapture
    if "username" in session:
        if request.method == "POST":
            basedir = os.path.abspath(os.path.dirname(__file__))
            # results = None
            if request.form.get('compareCam') == 'Compare!':
                names = request.form['selectName']
                filename = 'cameraCompare.png'
                directories = os.path.join(basedir, app.config["IMAGE_UPLOADS"], filename)
                results = compares(names, directories)
            else:
                image = request.files['image']
                print ("this is image", image)
                names = request.form['selectName']
                filename = 'uploadCompare.png'
                directories = os.path.join(basedir, app.config["IMAGE_UPLOADS"], filename)
                print("filename: ", directories)
                image.save(directories)
                print("Image saved")
                results = compares(names, directories)
            
            verifiedTotal = cursor.execute("SELECT document_verified FROM user WHERE username = %s", (username))
            verifiedTotal = cursor.fetchone()

            if results == True:
                verifiedTotal = verifiedTotal[0] + 1
                resultsInput = "MATCH"
            else:
                resultsInput = "FORGED"
                
            cmpTotal = cursor.execute("SELECT signature_comparison FROM user WHERE username = %s", (username))
            cmpTotal = cursor.fetchone()
            cmpTotal = cmpTotal[0] + 1
            cursor.execute("UPDATE user SET signature_comparison = %s , document_verified = %s WHERE username = %s", (cmpTotal, verifiedTotal , username))
            connection.commit()

            userID = cursor.execute("SELECT userID FROM user WHERE username = %s", (username))
            userID = cursor.fetchone()

            cursor.execute("INSERT INTO history (userID, comparingName, result) VALUES (%s, %s,%s)", (userID, names,resultsInput))
            connection.commit()

            if results == True:
                return redirect(url_for('success'))
            else:
                return redirect(url_for('fail'))
    else:
        return redirect(url_for('login'))

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/task',methods=['POST','GET'])
def tasks():
    if "username" in session:
        username = session['username']
        global switch,camera
        if request.method == 'POST':
            if request.form.get('click') == 'Capture':
                global capture
                capture=1
            elif  request.form.get('stop') == 'Stop/Start':
                if(switch==1):
                    switch=0
                    camera.release()
                    cv2.destroyAllWindows()
                else:
                    camera = cv2.VideoCapture(0)
                    switch=1
            elif  request.form.get('rec') == 'Start/Stop Recording':
                global rec, out
                rec= not rec
        elif request.method=='GET':
            return redirect(url_for('comparePhotos'))
        return redirect(url_for('comparePhotos'))
    else:
        return redirect(url_for('login'))

@app.route('/comparePhotos', methods=['GET', 'POST'])
def comparePhotos():
    if "username" in session:
        username = session['username']
        karyawan = cursor.execute('SELECT namaKaryawan FROM karyawan')
        karyawan = cursor.fetchall()
        global cameraOn, successCapture
        if cameraOn == 1:
            # time.sleep(5)
            cameraOn = 0
            while successCapture == 0:
                print('masih nyoba')
                pass
            # time.sleep(10)
            successCapture = 0
            stopcamera()
            return render_template('public/camera-confirm.html', usernames = username, karyawan = karyawan, len = len(karyawan))
    else:
        return redirect(url_for('login'))

@app.route('/home', methods = ['GET', 'POST'])
def home():
    if "username" in session:
        username = session['username']
        
        signature_registered = cursor.execute('SELECT signature_registered FROM user WHERE username = %s', username)
        signature_registered = cursor.fetchone()

        signature_comparison = cursor.execute('SELECT signature_comparison FROM user WHERE username = %s', username)
        signature_comparison = cursor.fetchone()

        document_verified = cursor.execute('SELECT document_verified FROM user WHERE username = %s', username)
        document_verified = cursor.fetchone()
        
        userID = cursor.execute('SELECT userID FROM user WHERE username = %s', username)
        userID = cursor.fetchone()
        history = cursor.execute('SELECT * FROM history WHERE userID = %s', userID)
        history = cursor.fetchall()

        karyawan = cursor.execute('SELECT namaKaryawan FROM karyawan')
        karyawan = cursor.fetchall()
        
        role = session['role']
        print(role)
        if role == 1:
            return redirect(url_for('admin_dashboard'))
        else:
            return render_template('public/dashboardUsers.html', sgn = signature_registered[0], sgn_cmp = signature_comparison[0], doc = document_verified[0], usernames = username, history = history,len = len(history), karyawan = karyawan, lenKaryawan = len(karyawan))
    else:
        return redirect(url_for('login'))
        
@app.route('/',methods=['GET','POST'])
def login():
    msg = ''
    if request.method == 'POST':
        nip = request.form['nip']

        username1 = cursor.execute("SELECT username FROM user WHERE nip = %s", (nip))
        username1 = cursor.fetchone()
        username = username1[0]
        
        password = request.form['password']
        cursor.execute('SELECT * FROM user WHERE username = %s AND passwords = %s', (username, password))
        account = cursor.fetchone()

        role = cursor.execute('SELECT is_admin FROM user WHERE username = %s AND passwords = %s', (username, password))
        role = cursor.fetchone()

        session['username'] = username
        session['role'] = role[0]
        if account:
            session['loggedin'] = True
            if role[0] == 1:
                return redirect(url_for('admin_dashboard', username = username))
            else:
                print('masuk')
                return redirect(url_for('home', username = username,**request.args))
        else:
            msg = 'Incorrect username/password!'
    return render_template('public/index.html',msg=msg)

@app.route('/logout')
def logout():
    session.pop('username',None)
    return render_template('public/index.html')

