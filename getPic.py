#coding:utf-8
import cv2, os, time
import numpy as np
import random

camera_path = '/dev/video11'

class camera:
    focus = int()
    score = float()
    weight = int()
    height = int()
    lr = int()
    gamma = int()
    # EarlyStop epoch 8
    stopper = list()
    best = float()
    epoch = int()
    max_epoch = int()
    best_focus = int()
    increase = float()
    F = 1
    
    def __init__(self):
        self.focus = 300
        self.score = 0.0
        self.weight, self.height = 3, 4
        self.lr = 10
        self.gamma = 5
        # EarlyStop epoch 8
        self.stopper = [1,1,1,1,1,1,1,1]
        self.best = 0.
        self.epoch = 0
        self.max_epoch = 50
        self.best_focus = self.focus
        self.increase = self.lr

        self.cap = cv2.VideoCapture(camera_path)
        self.F = 1
        
        if(self.cap.isOpened()):
            self.cap.set(self.weight, 1024)
            self.cap.set(self.height, 768) 
            
            print('Camera({}) initialze OK'.format(camera_path))
        else:
            print('Camera({}) initialze failed'.format(camera_path))
                    
    def __del__(self):
        if(self.cap.isOpened()):
            self.cap.release()
            try:
                cv2.destroyAllWindows()
            except Exception as e:
                print(e)  
            print('Camera({}) detroy OK'.format(camera_path))   
        else:
            print('Camera({}) detroy failed'.format(camera_path))
            
    def culculate(self,image):
        # variance: gradient amplitude
        # result more large, commonly more fittable
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        result = np.var(gradient_magnitude)
        return result

    def lr_gamma(self, lr, epoch):
        if(epoch < self.gamma):
            return lr
        else:
            return lr*(0.99**(epoch-self.gamma))

    def refocus(self):
        self.epoch = 0
        if(self.cap.isOpened()):
            # Randomized initialization
            lst = [0, self.focus/4, self.focus/2, 3*self.focus/4]
            for i in range(len(lst)):
                lst[i] += random.random()*self.focus/4
            scr = [0,0,0,0]
            for i in range(len(lst)):
                self.cap.set(cv2.CAP_PROP_FOCUS, lst[i])
                ret, frame = self.cap.read()
                scr[i] = self.culculate(frame)
                
            self.focus = lst[scr.index(max(scr))]
                
            # initialize
            self.cap.set(cv2.CAP_PROP_FOCUS, self.focus)
            ret, frame = self.cap.read()
            
            self.best = 0.0
            self.best = self.culculate(frame)
            self.score = self.best
            # First, we increase focus to test
            current_focus = self.focus + self.lr_gamma(self.lr, self.epoch)
            self.cap.set(cv2.CAP_PROP_FOCUS, current_focus)
            
            while True:
                ret, frame = self.cap.read()

                rNew = self.culculate(frame)
                
                # best should increase
                if (rNew - self.score) > 0:
                    self.stopper[self.epoch%8] = 1
                    # update measure
                    self.score = rNew
                    self.F = 1
                    
                    # move direction
                    if self.increase < 0:
                        self.increase = -self.lr_gamma(self.lr, self.epoch)
                    else:
                        self.increase = self.lr_gamma(self.lr, self.epoch)
                    self.focus = current_focus
                    
                    # best upload
                    if rNew > self.best:
                        self.best_focus = current_focus
                        
                    current_focus += self.increase
                else:
                    self.stopper[self.epoch%8] = 0
                    
                    self.F = 0
            
                    # move direction reverse
                    if self.increase < 0:
                        self.increase = self.lr_gamma(self.lr, self.epoch)
                    else:
                        self.increase = -self.lr_gamma(self.lr, self.epoch)
                        
                    current_focus = self.focus + self.increase
                
                if (not (1 in self.stopper)) or (self.epoch > self.max_epoch):
                    print("Auto focus OK.")
                    if self.epoch > self.max_epoch:
                        print("Have not restrain.")
                    return
                    
                self.epoch += 1
                
                print('Epoch: {}, Focus: {:.1f}, Focus Measure: {:.4f} '.format(
                    self.epoch, self.focus, rNew), '*' if self.F == 1 else '')
                
                self.cap.set(cv2.CAP_PROP_FOCUS, current_focus)
        else:
            print('Camera({}) focus failed'.format(camera_path))
            
        print('Camera({}) focus OK'.format(camera_path))

    def downloadPic(self):
        if(self.cap.isOpened()):
            self.cap.set(cv2.CAP_PROP_FOCUS, self.best_focus)
            ret, frame = self.cap.read()
            
            cv2.imwrite('./test.jpg', frame)
            os.system('chmod -R 777 test.jpg && xdg-open test.jpg')
            print("Best focus: {}".format(self.best_focus))
            
        else:
            print('Camera({}) download failed'.format(camera_path))

if __name__ == '__main__':
    
    _start = time.time()

    cam = camera()
    cam.refocus()
    cam.downloadPic()
    
    _end = time.time()
    
    print('Picture save OK\nTime: '+ str(_end - _start) +'\nexit program.')
    
