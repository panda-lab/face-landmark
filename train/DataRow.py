# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 22:30:29 2015

@author: Ishay Tubi
"""

import os
import cv2
import numpy as np
import sys
import csv
import time
def getGitRepFolder():
#    import subprocess
#    return subprocess.Popen(['git', 'rev-parse', '--show-toplevel'], stdout=subprocess.PIPE).communicate()[0].rstrip()
    return '/home/ly/workspace/Vanilla'

def mse_normlized(groundTruth, pred):
    delX = groundTruth[78]-groundTruth[84] 
    delY = groundTruth[79]-groundTruth[85] 
    interOc = (1e-6+(delX*delX + delY*delY))**0.5  # Euclidain distance
    diff = (pred-groundTruth)**2
    sumPairs = (diff[0::2]+diff[1::2])**0.5  # Euclidian distance 
    return (sumPairs / interOc)  # normlized 




class RetVal:
    pass  ## A generic class to return multiple values without a need for a dictionary.

def createDataRowsFromCSV(csvFilePath, csvParseFunc, DATA_PATH, limit = sys.maxint):
    ''' Returns a list of DataRow from CSV files parsed by csvParseFunc, 
        DATA_PATH is the prefix to add to the csv file names,
        limit can be used to parse only partial file rows.
    ''' 
    data = []  # the array we build
    validObjectsCounter = 0 
    
    with open(csvFilePath, 'r') as csvfile:

        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            d = csvParseFunc(row, DATA_PATH)
            if d is not None:
                data.append(d)
                validObjectsCounter += 1
                if (validObjectsCounter > limit ):  # Stop if reached to limit
                    return data 
    return data

def createBoxRowsFromCSV(csvFilePath, DATA_PATH, limit = sys.maxint):
    ''' Returns a list of DataRow from CSV files parsed by csvParseFunc, 
        DATA_PATH is the prefix to add to the csv file names,
        limit can be used to parse only partial file rows.
    ''' 
    data = []  # the array we build
    validObjectsCounter = 0 
    with open(csvFilePath, 'r') as csvfile:

        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            # print "row: ", row
            box = BBox()
            box.path = os.path.join(DATA_PATH, row[0]).replace("\\", "/")
            #print box.path
            box.left = row[1]
            box.top = row[2]
            box.right = row[3]
            box.bottom = row[4]
            if  box.right>0 and  box.bottom>0 :
                data.append(box)
                validObjectsCounter += 1
                if (validObjectsCounter > limit ):  # Stop if reached to limit
                    return data 
    return data

def getValidWithBBox(dataRows, boxRows=[]):
    ''' Returns a list of valid DataRow of a given list of dataRows 
    '''
    import dlib
    import random
    R=RetVal()
    
    R.outsideLandmarks = 0 
    R.noImages = 0 
    R.noFacesAtAll = 0 
    R.couldNotMatch = 0
    detector=dlib.get_frontal_face_detector()

    validRow=[]
    for rowIdx, dataRow in enumerate(dataRows):
        if dataRow.image is None or len(dataRow.image)==0:
            R.noImages += 1
            continue
        #print 'landmarks ', dataRow.landmarks()
        lmd_xy = dataRow.landmarks().reshape([-1,2])

        #left,  top = lmd_xy.min( axis=0 )
        #right, bot = lmd_xy.max( axis=0 )
        #left,  top = lmd_xy[37]
        #right, bot = lmd_xy[46]
        left,  top0 = lmd_xy[36]
        right, top1 = lmd_xy[45]
        mid, bot = lmd_xy[57]

        top = max([top0, top1])
        left = min([left, mid])
        right = max([right, mid])
        #left,  top = lmd_xy.min(axis=0)
        #right, bot = lmd_xy.max(axis=0)
        #print "landmark box:  ", left, top, right , bot


        if not boxRows == []:
            dets = [boxRows[rowIdx]]
            #print "box:  ",boxRows[rowIdx].left, boxRows[rowIdx].top, boxRows[rowIdx].right, boxRows[rowIdx].bottom
        else:
            dets = detector( np.array(dataRow.image, dtype = 'uint8' ), 1);
            #dets = [BBox.BBoxFromLTRB(left, top, right, bot)]


        det_bbox = None  # the valid bbox if found 
        #print R.couldNotMatch
        for det in dets:
            if not boxRows == []:
                det_box = BBox.BBoxFromLTRB(float(det.left), float(det.top), float(det.right), float(det.bottom))
                # print "det_box:  ",det_box.left, det_box.top, det_box.right, det_box.bottom
            else:
                det_box = BBox.BBoxFromLTRB(det.left(), det.top(), det.right(), det.bottom())
                #print det.left(), det.top(), det.right(), det.bottom()
                #det_box = BBox.BBoxFromLTRB(det[0], det[1], det[0]+det[2], det[1]+det[3])
                #det_box = det

            # Does all landmarks fit into this box?
            if top >= det_box.top and bot<= det_box.bottom and left>=det_box.left and right<=det_box.right:
                    # det_bbox = det_box
                    # height = det_box.bottom - det_box.top

                    # det_bbox.top = det_box.top - height * 0.1
                    # det_bbox.bottom = det_box.bottom + height*0.25
                    # weight = det_box.right - det_box.left
                    # det_bbox.left = det_box.left - weight*0.15 
                    # det_bbox.right = det_box.right + weight*0.15

                    # center random shift
                    tx = random.uniform(-0.04,0.04)
                    ty = random.uniform(-0.04,0.04)
                    det_box.offset(tx*det_box.width(), ty*det_box.height())

                    # scale random
                    s = random.uniform(-0.04,0.04)
                    #s = 0.1#random.uniform(-0.05,0.1)
                    #s = random.uniform(0.0,0.03)
                    det_bbox = det_box
                    height = det_box.bottom - det_box.top
                    det_bbox.top = det_box.top - height * s#0.1
                    det_bbox.bottom = det_box.bottom + height*s#0.1
                    weight = det_box.right - det_box.left
                    det_bbox.left = det_box.left - weight*s#0.1
                    det_bbox.right = det_box.right + weight*s#0.1
                    # print "det_bbox:  ",det_bbox.left, det_bbox.top, det_bbox.right, det_bbox.bottom
                    
        if det_bbox is None:
            if len(dets)>0:
                R.couldNotMatch += 1  # For statistics, dlib found faces but they did not match our landmarks.
            else:
                R.noFacesAtAll += 1  # dlib found 0 faces.
        else:
            dataRow.fbbox = det_bbox  # Save the bbox to the data row
            #if det_bbox.left<0 or det_bbox.top<0 or det_bbox.right>dataRow.image.shape[0] or det_bbox.bottom>dataRow.image.shape[1]:
            if det_bbox.left<0 or det_bbox.top<0 or det_bbox.right>dataRow.image.shape[1] or det_bbox.bottom>dataRow.image.shape[0]:
                #print 'det_bbox: ', det_bbox
                #print 'detb: ', detb
                #print 'image.shape: ', dataRow.image.shape
                R.outsideLandmarks += 1  # Saftey check, make sure nothing goes out of bound.
            else:
                validRow.append(dataRow)  
    
    
    return validRow,R 
        
def writeHD5(dataRows, outputPath, setTxtFilePATH, meanTrainSet, stdTrainSet , IMAGE_SIZE=60, mirror=False):
    ''' Create HD5 data set for caffe from given valid data rows.
    if mirror is True, duplicate data by mirroring. 
    ''' 
    from numpy import zeros
    import h5py
    
    if mirror:
        BATCH_SIZE = len(dataRows) *2
    else:
        BATCH_SIZE = len(dataRows) 

    #HD5Images = zeros([BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE], dtype='float32')
    HD5Images = zeros([BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE], dtype='float32')
    HD5Landmarks = zeros([BATCH_SIZE, 136], dtype='float32')
    #prefix  = os.path.join(ROOT, 'caffeData', 'hd5', 'train')
    setTxtFile = open(setTxtFilePATH, 'w')

        
    i = 0 
    
    for dataRowOrig in dataRows:
        if i % 1000 == 0 or i >= BATCH_SIZE-1:
            print "Processing row %d " % (i+1) 
            
        if not hasattr(dataRowOrig, 'fbbox'):
            print "Warning, no fbbox"
            continue
        
        dataRow = dataRowOrig.copyCroppedByBBox(dataRowOrig.fbbox)  # Get a cropped scale copy of the data row
        scaledLM = dataRow.landmarksScaledMinus05_plus05() 
        image = dataRow.image.astype('f4')
        #image = (image-meanTrainSet)/(1.e-6 + stdTrainSet)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).reshape(1,IMAGE_SIZE,IMAGE_SIZE)
        m, s = cv2.meanStdDev(image)
        image = (image-m)/(1.e-6 + s)
        
        #HD5Images[i, :] = cv2.split(image)  # split interleaved (40,40,3) to (3,40,40)
        HD5Images[i, :] = image
        HD5Landmarks[i,:] = scaledLM
        i+=1
        
        if mirror:
            dataRow = dataRowOrig.copyCroppedByBBox(dataRowOrig.fbbox).copyMirrored()  # Get a cropped scale copy of the data row
            scaledLM = dataRow.landmarksScaledMinus05_plus05() 
            image = dataRow.image.astype('f4')
            #image = (image-meanTrainSet)/(1.e-6 + stdTrainSet)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).reshape(1,IMAGE_SIZE,IMAGE_SIZE)
            m, s = cv2.meanStdDev(image)
            image = (image-m)/(1.e-6 + s)
            
            #HD5Images[i, :] = cv2.split(image)  # split interleaved (40,40,3) to (3,40,40)
            HD5Images[i, :] = image
            HD5Landmarks[i,:] = scaledLM
            i+=1
        
        
    with h5py.File(outputPath, 'w') as T:
        T.create_dataset("X", data=HD5Images)
        T.create_dataset("landmarks", data=HD5Landmarks)

    setTxtFile.write(outputPath+"\n")
    setTxtFile.flush()
    setTxtFile.close()
    
    
    
  



class ErrorAcum:  # Used to count error per landmark
    def __init__(self):
        self.errorPerLandmark = np.zeros(68, dtype ='f4')
        self.itemsCounter = 0
        self.failureCounter = 0
        
    def __repr__(self):
        return '%f mean error, %d items, %d failures  %f precent' % (self.meanError().mean()*100, self.itemsCounter, self.failureCounter, float(self.failureCounter)/self.itemsCounter if self.itemsCounter>0 else 0)
        
        
    def add(self, groundTruth, pred):
        normlized = mse_normlized(groundTruth, pred)
        self.errorPerLandmark += normlized
        self.itemsCounter +=1
        if normlized.mean() > 0.1: 
            # Count error above 10% as failure
            self.failureCounter +=1

    def meanError(self):
        if self.itemsCounter > 0:
            return self.errorPerLandmark/self.itemsCounter
        else:
            return self.errorPerLandmark

    def __add__(self, x):
        ret = ErrorAcum()
        ret.errorPerLandmark = self.errorPerLandmark + x.errorPerLandmark
        ret.itemsCounter    = self.itemsCounter + x.itemsCounter
        ret.failureCounter  = self.failureCounter + x.failureCounter        
        return ret
        
    def plot(self):
        from matplotlib.pylab import show, plot, stem
        pass


class BBox:  # Bounding box
    
    @staticmethod
    def BBoxFromLTRB(l, t, r, b):
        return BBox(l, t, r, b)
    
    @staticmethod
    def BBoxFromXYWH_array(xywh):
        return BBox(xywh[0], xywh[1], +xywh[0]+xywh[2], xywh[1]+xywh[3])
    
    @staticmethod
    def BBoxFromXYWH(x,y,w,h):
        return BBox(x,y, x+w, y+h)
    
    def top_left(self):
        return (self.top, self.left)
    
    def left_top(self):
        return (self.left, self.top)

    def bottom_right(self):
        return (self.bottom, self.right)

    def right_bottom(self):
        return (self.right, self.bottom)
    
    def right_top(self):
        return (self.right, self.top)
    
    def relaxed(self, clip ,relax=3):  #@Unused
        from numpy import array
        _A = array
        maxWidth, maxHeight =  clip[0], clip[1]
        
        nw, nh = self.size()*(1+relax)*.5       
        center = self.center()
        offset=_A([nw,nh])
        lefttop = center - offset
        rightbot= center + offset 
         
        self.left, self.top  = int( max( 0, lefttop[0] ) ), int( max( 0, lefttop[1]) )
        self.right, self.bottom = int( min( rightbot[0], maxWidth ) ), int( min( rightbot[1], maxHeight ) )
        return self

    def clip(self, maxRight, maxBottom):
        self.left = max(self.left, 0)
        self.top = max(self.top, 0)
        self.right = min(self.right, maxRight)
        self.bottom = min(self.bottom, maxBottom)
        
    def size(self):
        from numpy import  array
        return array([self.width(), self.height()])
     
    def center(self):
        from numpy import  array
        return array([(self.left+self.right)/2, (self.top+self.bottom)/2])
                
    def __init__(self,left=0, top=0, right=0, bottom=0, path=''):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.path = path
        
    def width(self):
        return self.right - self.left
        
    def height(self):
        return self.bottom - self.top
        
    def xywh(self):
        return self.left, self.top, self.width(), self.height()
        
    def offset(self, x, y):
        self.left += x 
        self.right += x
        self.top += y 
        self.bottom += y
         
    def scale(self, rx, ry):
        self.left *= rx 
        self.right *= rx
        self.top *= ry 
        self.bottom *= ry
                        
    def __repr__(self):
        return 'left(%.1f), top(%.1f), right(%.1f), bottom(%.1f) w(%d) h(%d)' % (self.left, self.top, self.right, self.bottom,self.width(), self.height())

    def makeInt(self):
        self.left    = int(self.left)
        self.top     = int(self.top)
        self.right   = int(self.right)
        self.bottom  = int(self.bottom)
        return self



class DataRow:
    global TrainSetMean
    global TrainSetSTD
    
    IMAGE_SIZE = 60
    def __init__(self, path='', p1=(0, 0, ), p2=(0, 0), p3=(0, 0), p4=(0, 0), p5=(0, 0),p6=(0, 0, ), p7=(0, 0), p8=(0, 0), p9=(0, 0), p10=(0, 0), 
p11=(0, 0, ), p12=(0, 0), p13=(0, 0), p14=(0, 0), p15=(0, 0),p16=(0, 0, ), p17=(0, 0), p18=(0, 0), p19=(0, 0), p20=(0, 0),
p21=(0, 0, ), p22=(0, 0), p23=(0, 0), p24=(0, 0), p25=(0, 0),p26=(0, 0, ), p27=(0, 0), p28=(0, 0), p29=(0, 0), p30=(0, 0),
p31=(0, 0, ), p32=(0, 0), p33=(0, 0), p34=(0, 0), p35=(0, 0),p36=(0, 0, ), p37=(0, 0), p38=(0, 0), p39=(0, 0), p40=(0, 0),
p41=(0, 0, ), p42=(0, 0), p43=(0, 0), p44=(0, 0), p45=(0, 0),p46=(0, 0, ), p47=(0, 0), p48=(0, 0), p49=(0, 0), p50=(0, 0),
p51=(0, 0, ), p52=(0, 0), p53=(0, 0), p54=(0, 0), p55=(0, 0),p56=(0, 0, ), p57=(0, 0), p58=(0, 0), p59=(0, 0), p60=(0, 0),
p61=(0, 0, ), p62=(0, 0), p63=(0, 0), p64=(0, 0), p65=(0, 0),p66=(0, 0, ), p67=(0, 0), p68=(0, 0)):
        self.image = cv2.imread(path)
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.p5 = p5
        self.p6 = p6
        self.p7 = p7
        self.p8 = p8
        self.p9 = p9
        self.p10 = p10
        self.p11 = p11
        self.p12 = p12
        self.p13 = p13
        self.p14 = p14
        self.p15 = p15
        self.p16 = p16
        self.p17 = p17
        self.p18 = p18
        self.p19 = p19
        self.p20 = p20
        self.p21 = p21
        self.p22 = p22
        self.p23 = p23
        self.p24 = p24
        self.p25 = p25
        self.p26 = p26
        self.p27 = p27
        self.p28 = p28
        self.p29 = p29
        self.p30 = p30
        self.p31 = p31
        self.p32 = p32
        self.p33 = p33
        self.p34 = p34
        self.p35 = p35
        self.p36 = p36
        self.p37 = p37
        self.p38 = p38
        self.p39 = p39
        self.p40 = p40
        self.p41 = p41
        self.p42 = p42
        self.p43 = p43
        self.p44 = p44
        self.p45 = p45
        self.p46 = p46
        self.p47 = p47
        self.p48 = p48
        self.p49 = p49
        self.p50 = p50
        self.p51 = p51
        self.p52 = p52
        self.p53 = p53
        self.p54 = p54
        self.p55 = p55
        self.p56 = p56
        self.p57 = p57
        self.p58 = p58
        self.p59 = p59
        self.p60 = p60
        self.p61 = p61
        self.p62 = p62
        self.p63 = p63
        self.p64 = p64
        self.p65 = p65
        self.p66 = p66
        self.p67 = p67
        self.p68 = p68

        self.name = os.path.split(path)[-1]
        self.sx = 1.
        self.sy = 1.
        self.offsetX = 0.
        self.offsetY = 0.

    def __repr__(self):
        return '{} le:{},{} re:{},{} nose:{},{}, lm:{},{} rm:{},{}'.format(
            self.name,
            self.p1[0], self.p1[1],
            self.p2[0], self.p2[1],
            self.p3[0], self.p3[1],
            self.p4[0], self.p4[1],
            self.p5[0], self.p5[1],
            self.p6[0], self.p6[1],
            self.p7[0], self.p7[1],
            self.p8[0], self.p8[1],
            self.p9[0], self.p9[1],
            self.p10[0], self.p10[1],
            self.p11[0], self.p11[1],
            self.p12[0], self.p12[1],
            self.p13[0], self.p13[1],
            self.p14[0], self.p14[1],
            self.p15[0], self.p15[1],
            self.p16[0], self.p16[1],
            self.p17[0], self.p17[1],
            self.p18[0], self.p18[1],
            self.p19[0], self.p19[1],
            self.p20[0], self.p20[1],
            self.p21[0], self.p21[1],
            self.p22[0], self.p22[1],
            self.p23[0], self.p23[1],
            self.p24[0], self.p24[1],
            self.p25[0], self.p25[1],
            self.p26[0], self.p26[1],
            self.p27[0], self.p27[1],
            self.p28[0], self.p28[1],
            self.p29[0], self.p29[1],
            self.p30[0], self.p20[1],
            self.p31[0], self.p31[1],
            self.p32[0], self.p32[1],
            self.p33[0], self.p33[1],
            self.p34[0], self.p34[1],
            self.p35[0], self.p35[1],
            self.p36[0], self.p36[1],
            self.p37[0], self.p37[1],
            self.p38[0], self.p38[1],
            self.p39[0], self.p39[1],
            self.p40[0], self.p40[1],
            self.p51[0], self.p51[1],
            self.p52[0], self.p52[1],
            self.p53[0], self.p53[1],
            self.p54[0], self.p54[1],
            self.p55[0], self.p55[1],
            self.p56[0], self.p56[1],
            self.p57[0], self.p57[1],
            self.p58[0], self.p58[1],
            self.p59[0], self.p59[1],
            self.p60[0], self.p60[1],
            self.p61[0], self.p61[1],
            self.p62[0], self.p62[1],
            self.p63[0], self.p63[1],
            self.p64[0], self.p64[1],
            self.p65[0], self.p65[1],
            self.p66[0], self.p66[1],
            self.p67[0], self.p67[1],
            self.p68[0], self.p68[1]
            )

    def setLandmarks(self,landMarks):
        """
        @landMarks : np.array
        set the landmarks from array
        """
        self.p1 = landMarks[0:2]
        self.p2 = landMarks[2:4]
        self.p3 = landMarks[4:6]
        self.p4 = landMarks[6:8]
        self.p5 = landMarks[8:10]
        self.p6 = landMarks[10:12]
        self.p7 = landMarks[12:14]
        self.p8 = landMarks[14:16]
        self.p9 = landMarks[16:18]
        self.p10 = landMarks[18:20]
        self.p11 = landMarks[20:22]
        self.p12 = landMarks[22:24]
        self.p13 = landMarks[24:26]
        self.p14 = landMarks[26:28]
        self.p15 = landMarks[28:30]
        self.p16 = landMarks[30:32]
        self.p17 = landMarks[32:34]
        self.p18 = landMarks[34:36]
        self.p19 = landMarks[36:38]
        self.p20 = landMarks[38:40]
        self.p21 = landMarks[40:42]
        self.p22 = landMarks[42:44]
        self.p23 = landMarks[44:46]
        self.p24 = landMarks[46:48]
        self.p25 = landMarks[48:50]
        self.p26 = landMarks[50:52]
        self.p27 = landMarks[52:54]
        self.p28 = landMarks[54:56]
        self.p29 = landMarks[56:58]
        self.p30 = landMarks[58:60]
        self.p31 = landMarks[60:62]
        self.p32 = landMarks[62:64]
        self.p33 = landMarks[64:66]
        self.p34 = landMarks[66:68]
        self.p35 = landMarks[68:70]
        self.p36 = landMarks[70:72]
        self.p37 = landMarks[72:74]
        self.p38 = landMarks[74:76]
        self.p39 = landMarks[76:78]
        self.p40 = landMarks[78:80]        
        self.p41 = landMarks[80:82]
        self.p42 = landMarks[82:84]
        self.p43 = landMarks[84:86]
        self.p44 = landMarks[86:88]
        self.p45 = landMarks[88:90]
        self.p46 = landMarks[90:92]
        self.p47 = landMarks[92:94]
        self.p48 = landMarks[94:96]
        self.p49 = landMarks[96:98]
        self.p50 = landMarks[98:100]
        self.p51 = landMarks[100:102]
        self.p52 = landMarks[102:104]
        self.p53 = landMarks[104:106]
        self.p54 = landMarks[106:108]
        self.p55 = landMarks[108:110]
        self.p56 = landMarks[110:112]
        self.p57 = landMarks[112:114]
        self.p58 = landMarks[114:116]
        self.p59 = landMarks[116:118]
        self.p60 = landMarks[118:120]
        self.p61 = landMarks[120:122]
        self.p62 = landMarks[122:124]
        self.p63 = landMarks[124:126]
        self.p64 = landMarks[126:128]
        self.p65 = landMarks[128:130]
        self.p66 = landMarks[130:132]
        self.p67 = landMarks[132:134]
        self.p68 = landMarks[134:136]       
    def landmarks(self):
        # return numpy float array with ordered values
        stright = [
            self.p1[0],self.p1[1],
            self.p2[0],self.p2[1],
            self.p3[0],self.p3[1],
            self.p4[0],self.p4[1],
            self.p5[0],self.p5[1],
            self.p6[0],self.p6[1],
            self.p7[0],self.p7[1],
            self.p8[0],self.p8[1],
            self.p9[0],self.p9[1],
            self.p10[0],self.p10[1],
            self.p11[0],self.p11[1],
            self.p12[0],self.p12[1],
            self.p13[0],self.p13[1],
            self.p14[0],self.p14[1],
            self.p15[0],self.p15[1],
            self.p16[0],self.p16[1],
            self.p17[0],self.p17[1],
            self.p18[0],self.p18[1],
            self.p19[0],self.p19[1],
            self.p20[0],self.p20[1],
            self.p21[0],self.p21[1],
            self.p22[0],self.p22[1],
            self.p23[0],self.p23[1],
            self.p24[0],self.p24[1],
            self.p25[0],self.p25[1],
            self.p26[0],self.p26[1],
            self.p27[0],self.p27[1],
            self.p28[0],self.p28[1],
            self.p29[0],self.p29[1],
            self.p30[0],self.p30[1],
            self.p31[0],self.p31[1],
            self.p32[0],self.p32[1],
            self.p33[0],self.p33[1],
            self.p34[0],self.p34[1],
            self.p35[0],self.p35[1],
            self.p36[0],self.p36[1],
            self.p37[0],self.p37[1],
            self.p38[0],self.p38[1],
            self.p39[0],self.p39[1],
            self.p40[0],self.p40[1],
            self.p41[0],self.p41[1],
            self.p42[0],self.p42[1],
            self.p43[0],self.p43[1],
            self.p44[0],self.p44[1],
            self.p45[0],self.p45[1],
            self.p46[0],self.p46[1],
            self.p47[0],self.p47[1],
            self.p48[0],self.p48[1],
            self.p49[0],self.p49[1],
            self.p50[0],self.p50[1],
            self.p51[0],self.p51[1],
            self.p52[0],self.p52[1],
            self.p53[0],self.p53[1],
            self.p54[0],self.p54[1],
            self.p55[0],self.p55[1],
            self.p56[0],self.p56[1],
            self.p57[0],self.p57[1],
            self.p58[0],self.p58[1],
            self.p59[0],self.p59[1],
            self.p60[0],self.p60[1],
            self.p61[0],self.p61[1],
            self.p62[0],self.p62[1],
            self.p63[0],self.p63[1],
            self.p64[0],self.p64[1],
            self.p65[0],self.p65[1],
            self.p66[0],self.p66[1],
            self.p67[0],self.p67[1],
            self.p68[0],self.p68[1]]

        return np.array(stright, dtype='f4')

    def landmarksScaledMinus05_plus05(self):
        # return numpy float array with ordered values
        #return self.landmarks().astype('f4')/40. - 0.5
        return self.landmarks().astype('f4')/60.
        
    def scale(self, sx, sy):
        self.sx *= sx
        self.sy *= sy

        self.p1 = (self.p1[0]*sx, self.p1[1]*sy)
        self.p2 = (self.p2[0]*sx, self.p2[1]*sy)
        self.p3 = (self.p3[0]*sx, self.p3[1]*sy)
        self.p4 = (self.p4[0]*sx, self.p4[1]*sy)
        self.p5 = (self.p5[0]*sx, self.p5[1]*sy)
        self.p6 = (self.p6[0]*sx, self.p6[1]*sy)
        self.p7 = (self.p7[0]*sx, self.p7[1]*sy)
        self.p8 = (self.p8[0]*sx, self.p8[1]*sy)
        self.p9 = (self.p9[0]*sx, self.p9[1]*sy)
        self.p10 = (self.p10[0]*sx, self.p10[1]*sy)
        self.p11 = (self.p11[0]*sx, self.p11[1]*sy)
        self.p12 = (self.p12[0]*sx, self.p12[1]*sy)
        self.p13 = (self.p13[0]*sx, self.p13[1]*sy)
        self.p14 = (self.p14[0]*sx, self.p14[1]*sy)
        self.p15 = (self.p15[0]*sx, self.p15[1]*sy)
        self.p16 = (self.p16[0]*sx, self.p16[1]*sy)
        self.p17 = (self.p17[0]*sx, self.p17[1]*sy)
        self.p18 = (self.p18[0]*sx, self.p18[1]*sy)
        self.p19 = (self.p19[0]*sx, self.p19[1]*sy)
        self.p20 = (self.p20[0]*sx, self.p20[1]*sy)
        self.p21 = (self.p21[0]*sx, self.p21[1]*sy)
        self.p22 = (self.p22[0]*sx, self.p22[1]*sy)
        self.p23 = (self.p23[0]*sx, self.p23[1]*sy)
        self.p24 = (self.p24[0]*sx, self.p24[1]*sy)
        self.p25 = (self.p25[0]*sx, self.p25[1]*sy)
        self.p26 = (self.p26[0]*sx, self.p26[1]*sy)
        self.p27 = (self.p27[0]*sx, self.p27[1]*sy)
        self.p28 = (self.p28[0]*sx, self.p28[1]*sy)
        self.p29 = (self.p29[0]*sx, self.p29[1]*sy)
        self.p30 = (self.p30[0]*sx, self.p30[1]*sy)
        self.p31 = (self.p31[0]*sx, self.p31[1]*sy)
        self.p32 = (self.p32[0]*sx, self.p32[1]*sy)
        self.p33 = (self.p33[0]*sx, self.p33[1]*sy)
        self.p34 = (self.p34[0]*sx, self.p34[1]*sy)
        self.p35 = (self.p35[0]*sx, self.p35[1]*sy)
        self.p36 = (self.p36[0]*sx, self.p36[1]*sy)
        self.p37 = (self.p37[0]*sx, self.p37[1]*sy)
        self.p38 = (self.p38[0]*sx, self.p38[1]*sy)
        self.p39 = (self.p39[0]*sx, self.p39[1]*sy)
        self.p40 = (self.p40[0]*sx, self.p40[1]*sy)
        self.p41 = (self.p41[0]*sx, self.p41[1]*sy)
        self.p42 = (self.p42[0]*sx, self.p42[1]*sy)
        self.p43 = (self.p43[0]*sx, self.p43[1]*sy)
        self.p44 = (self.p44[0]*sx, self.p44[1]*sy)
        self.p45 = (self.p45[0]*sx, self.p45[1]*sy)
        self.p46 = (self.p46[0]*sx, self.p46[1]*sy)
        self.p47 = (self.p47[0]*sx, self.p47[1]*sy)
        self.p48 = (self.p48[0]*sx, self.p48[1]*sy)
        self.p49 = (self.p49[0]*sx, self.p49[1]*sy)
        self.p50 = (self.p50[0]*sx, self.p50[1]*sy)
        self.p51 = (self.p51[0]*sx, self.p51[1]*sy)
        self.p52 = (self.p52[0]*sx, self.p52[1]*sy)
        self.p53 = (self.p53[0]*sx, self.p53[1]*sy)
        self.p54 = (self.p54[0]*sx, self.p54[1]*sy)
        self.p55 = (self.p55[0]*sx, self.p55[1]*sy)
        self.p56 = (self.p56[0]*sx, self.p56[1]*sy)
        self.p57 = (self.p57[0]*sx, self.p57[1]*sy)
        self.p58 = (self.p58[0]*sx, self.p58[1]*sy)
        self.p59 = (self.p59[0]*sx, self.p59[1]*sy)
        self.p60 = (self.p60[0]*sx, self.p60[1]*sy)
        self.p61 = (self.p61[0]*sx, self.p61[1]*sy)
        self.p62 = (self.p62[0]*sx, self.p62[1]*sy)
        self.p63 = (self.p63[0]*sx, self.p63[1]*sy)
        self.p64 = (self.p64[0]*sx, self.p64[1]*sy)
        self.p65 = (self.p65[0]*sx, self.p65[1]*sy)
        self.p66 = (self.p66[0]*sx, self.p66[1]*sy)
        self.p67 = (self.p67[0]*sx, self.p67[1]*sy)
        self.p68 = (self.p68[0]*sx, self.p68[1]*sy)

        
        if hasattr(self, 'prediction'):
            self.prediction = self.prediction.reshape(-1, 2)*[sx, sy]

        return self

    def offsetCropped(self, offset=(0., 0.)):
        """ given the cropped values - offset the positions by offset
        """
        self.offsetX -= offset[0]
        self.offsetY -= offset[1]

        if hasattr(self, 'prediction'):
            self.prediction = self.prediction.reshape(-1,2)-offset


        self.p1 = (self.p1[0]-offset[0], self.p1[1]-offset[1])
        self.p2 = (self.p2[0]-offset[0], self.p2[1]-offset[1])
        self.p3 = (self.p3[0]-offset[0], self.p3[1]-offset[1])
        self.p4 = (self.p4[0]-offset[0], self.p4[1]-offset[1])
        self.p5 = (self.p5[0]-offset[0], self.p5[1]-offset[1])
        self.p6 = (self.p6[0]-offset[0], self.p6[1]-offset[1])
        self.p7 = (self.p7[0]-offset[0], self.p7[1]-offset[1])
        self.p8 = (self.p8[0]-offset[0], self.p8[1]-offset[1])
        self.p9 = (self.p9[0]-offset[0], self.p9[1]-offset[1])
        self.p10 = (self.p10[0]-offset[0], self.p10[1]-offset[1])
        self.p11 = (self.p11[0]-offset[0], self.p11[1]-offset[1])
        self.p12 = (self.p12[0]-offset[0], self.p12[1]-offset[1])
        self.p13 = (self.p13[0]-offset[0], self.p13[1]-offset[1])
        self.p14 = (self.p14[0]-offset[0], self.p14[1]-offset[1])
        self.p15 = (self.p15[0]-offset[0], self.p15[1]-offset[1])
        self.p16 = (self.p16[0]-offset[0], self.p16[1]-offset[1])
        self.p17 = (self.p17[0]-offset[0], self.p17[1]-offset[1])
        self.p18 = (self.p18[0]-offset[0], self.p18[1]-offset[1])
        self.p19 = (self.p19[0]-offset[0], self.p19[1]-offset[1])
        self.p20 = (self.p20[0]-offset[0], self.p20[1]-offset[1])
        self.p21 = (self.p21[0]-offset[0], self.p21[1]-offset[1])
        self.p22 = (self.p22[0]-offset[0], self.p22[1]-offset[1])
        self.p23 = (self.p23[0]-offset[0], self.p23[1]-offset[1])
        self.p24 = (self.p24[0]-offset[0], self.p24[1]-offset[1])
        self.p25 = (self.p25[0]-offset[0], self.p25[1]-offset[1])
        self.p26 = (self.p26[0]-offset[0], self.p26[1]-offset[1])
        self.p27 = (self.p27[0]-offset[0], self.p27[1]-offset[1])
        self.p28 = (self.p28[0]-offset[0], self.p28[1]-offset[1])
        self.p29 = (self.p29[0]-offset[0], self.p29[1]-offset[1])
        self.p30 = (self.p30[0]-offset[0], self.p30[1]-offset[1])
        self.p31 = (self.p31[0]-offset[0], self.p31[1]-offset[1])
        self.p32 = (self.p32[0]-offset[0], self.p32[1]-offset[1])
        self.p33 = (self.p33[0]-offset[0], self.p33[1]-offset[1])
        self.p34 = (self.p34[0]-offset[0], self.p34[1]-offset[1])
        self.p35 = (self.p35[0]-offset[0], self.p35[1]-offset[1])
        self.p36 = (self.p36[0]-offset[0], self.p36[1]-offset[1])
        self.p37 = (self.p37[0]-offset[0], self.p37[1]-offset[1])
        self.p38 = (self.p38[0]-offset[0], self.p38[1]-offset[1])
        self.p39 = (self.p39[0]-offset[0], self.p39[1]-offset[1])
        self.p40 = (self.p40[0]-offset[0], self.p40[1]-offset[1])
        self.p41 = (self.p41[0]-offset[0], self.p41[1]-offset[1])
        self.p42 = (self.p42[0]-offset[0], self.p42[1]-offset[1])
        self.p43 = (self.p43[0]-offset[0], self.p43[1]-offset[1])
        self.p44 = (self.p44[0]-offset[0], self.p44[1]-offset[1])
        self.p45 = (self.p45[0]-offset[0], self.p45[1]-offset[1])
        self.p46 = (self.p46[0]-offset[0], self.p46[1]-offset[1])
        self.p47 = (self.p47[0]-offset[0], self.p47[1]-offset[1])
        self.p48 = (self.p48[0]-offset[0], self.p48[1]-offset[1])
        self.p49 = (self.p49[0]-offset[0], self.p49[1]-offset[1])
        self.p50 = (self.p50[0]-offset[0], self.p50[1]-offset[1])
        self.p51 = (self.p51[0]-offset[0], self.p51[1]-offset[1])
        self.p52 = (self.p52[0]-offset[0], self.p52[1]-offset[1])
        self.p53 = (self.p53[0]-offset[0], self.p53[1]-offset[1])
        self.p54 = (self.p54[0]-offset[0], self.p54[1]-offset[1])
        self.p55 = (self.p55[0]-offset[0], self.p55[1]-offset[1])
        self.p56 = (self.p56[0]-offset[0], self.p56[1]-offset[1])
        self.p57 = (self.p57[0]-offset[0], self.p57[1]-offset[1])
        self.p58 = (self.p58[0]-offset[0], self.p58[1]-offset[1])
        self.p59 = (self.p59[0]-offset[0], self.p59[1]-offset[1])
        self.p60 = (self.p60[0]-offset[0], self.p60[1]-offset[1])
        self.p61 = (self.p61[0]-offset[0], self.p61[1]-offset[1])
        self.p62 = (self.p62[0]-offset[0], self.p62[1]-offset[1])
        self.p63 = (self.p63[0]-offset[0], self.p63[1]-offset[1])
        self.p64 = (self.p64[0]-offset[0], self.p64[1]-offset[1])
        self.p65 = (self.p65[0]-offset[0], self.p65[1]-offset[1])
        self.p66 = (self.p66[0]-offset[0], self.p66[1]-offset[1])
        self.p67 = (self.p67[0]-offset[0], self.p67[1]-offset[1])
        self.p68 = (self.p68[0]-offset[0], self.p68[1]-offset[1])

        return self

    def inverseScaleAndOffset(self, landmarks):
        """ computes the inverse scale and offset of input data according to the inverse scale factor and inverse offset factor
        """
        from numpy import array; _A = array ; # Shothand 
        
        ret = _A(landmarks.reshape(-1,2)) *_A([1./self.sx, 1./self.sy])
        ret += _A([-self.offsetX, -self.offsetY])
        return ret

    @staticmethod
    def DataRowFromNameBoxInterlaved(row, root=''):  # lfw_5590 + net_7876 (interleaved) 
        '''
        name , bounding box(w,h), left eye (x,y) ,right eye (x,y)..nose..left mouth,..right mouth
        '''
        d = DataRow()
        #print 'row: ', row
        d.path = os.path.join(root, row[0]).replace("\\", "/")
        print d.path
        d.name = os.path.split(d.path)[-1]

        d.image = cv2.imread(d.path)

        d.p1 = (float(row[1]), float(row[2]))
        d.p2 = (float(row[3]), float(row[4]))
        d.p3 = (float(row[5]), float(row[6]))
        d.p4 = (float(row[7]), float(row[8]))
        d.p5 = (float(row[9]), float(row[10]))
        d.p6 = (float(row[11]), float(row[12]))
        d.p7 = (float(row[13]), float(row[14]))
        d.p8 = (float(row[15]), float(row[16]))
        d.p9 = (float(row[17]), float(row[18]))
        d.p10 = (float(row[19]), float(row[20]))
        d.p11 = (float(row[21]), float(row[22]))
        d.p12 = (float(row[23]), float(row[24]))
        d.p13 = (float(row[25]), float(row[26]))
        d.p14 = (float(row[27]), float(row[28]))
        d.p15 = (float(row[29]), float(row[30]))
        d.p16 = (float(row[31]), float(row[32]))
        d.p17 = (float(row[33]), float(row[34]))
        d.p18 = (float(row[35]), float(row[36]))
        d.p19 = (float(row[37]), float(row[38]))
        d.p20 = (float(row[39]), float(row[40])) 
        d.p21 = (float(row[41]), float(row[42]))
        d.p22 = (float(row[43]), float(row[44]))
        d.p23 = (float(row[45]), float(row[46]))
        d.p24 = (float(row[47]), float(row[48]))
        d.p25 = (float(row[49]), float(row[50]))
        d.p26 = (float(row[51]), float(row[52]))
        d.p27 = (float(row[53]), float(row[54]))
        d.p28 = (float(row[55]), float(row[56]))
        d.p29 = (float(row[57]), float(row[58]))
        d.p30 = (float(row[59]), float(row[60])) 
        d.p31 = (float(row[61]), float(row[62]))
        d.p32 = (float(row[63]), float(row[64]))
        d.p33 = (float(row[65]), float(row[66]))
        d.p34 = (float(row[67]), float(row[68]))
        d.p35 = (float(row[69]), float(row[70]))
        d.p36 = (float(row[71]), float(row[72]))
        d.p37 = (float(row[73]), float(row[74]))
        d.p38 = (float(row[75]), float(row[76]))
        d.p39 = (float(row[77]), float(row[78]))
        d.p40 = (float(row[79]), float(row[80])) 
        d.p41 = (float(row[81]), float(row[82]))
        d.p42 = (float(row[83]), float(row[84]))
        d.p43 = (float(row[85]), float(row[86]))
        d.p44 = (float(row[87]), float(row[88]))
        d.p45 = (float(row[89]), float(row[90]))
        d.p46 = (float(row[91]), float(row[92]))
        d.p47 = (float(row[93]), float(row[94]))
        d.p48 = (float(row[95]), float(row[96]))
        d.p49 = (float(row[97]), float(row[98]))
        d.p50 = (float(row[99]), float(row[100]))
        d.p51 = (float(row[101]), float(row[102]))
        d.p52 = (float(row[103]), float(row[104]))
        d.p53 = (float(row[105]), float(row[106]))
        d.p54 = (float(row[107]), float(row[108]))
        d.p55 = (float(row[109]), float(row[110]))
        d.p56 = (float(row[111]), float(row[112]))
        d.p57 = (float(row[113]), float(row[114]))
        d.p58 = (float(row[115]), float(row[116]))
        d.p59 = (float(row[117]), float(row[118]))
        d.p60 = (float(row[119]), float(row[120]))
        d.p61 = (float(row[121]), float(row[122]))
        d.p62 = (float(row[123]), float(row[124]))
        d.p63 = (float(row[125]), float(row[126]))
        d.p64 = (float(row[127]), float(row[128]))
        d.p65 = (float(row[129]), float(row[130]))
        d.p66 = (float(row[131]), float(row[132]))
        d.p67 = (float(row[133]), float(row[134]))
        d.p68 = (float(row[135]), float(row[136]))

        return d

    @staticmethod
    def DataRowFromMTFL(row, root=''):
        '''
        --x1...x5,y1...y5: the locations for left eye, right eye, nose, left mouth corner, right mouth corner.
        '''
        d = DataRow()
        if len(row[0]) <= 1:
            # bug in the files, it has spaces seperating them, skip it
            row=row[1:]
            
        if len(row)<10:
            print 'error parsing ', row
            return None
        
        d.path = os.path.join(root, row[0]).replace("\\", "/")
        d.name = os.path.split(d.path)[-1]
        d.image = cv2.imread(d.path)
        
        if d.image is None:
            print 'Error reading image', d.path
            return None
        
        d.p1 = (float(row[1]), float(row[2]))
        d.p2 = (float(row[3]), float(row[4]))
        d.p3 = (float(row[5]), float(row[6]))
        d.p4 = (float(row[7]), float(row[8]))
        d.p5 = (float(row[9]), float(row[10]))
        d.p6 = (float(row[11]), float(row[12]))
        d.p7 = (float(row[13]), float(row[14]))
        d.p8 = (float(row[15]), float(row[16]))
        d.p9 = (float(row[17]), float(row[18]))
        d.p10 = (float(row[19]), float(row[20]))
        d.p11 = (float(row[21]), float(row[22]))
        d.p12 = (float(row[23]), float(row[24]))
        d.p13 = (float(row[25]), float(row[26]))
        d.p14 = (float(row[27]), float(row[28]))
        d.p15 = (float(row[29]), float(row[30]))
        d.p16 = (float(row[31]), float(row[32]))
        d.p17 = (float(row[33]), float(row[34]))
        d.p18 = (float(row[35]), float(row[36]))
        d.p19 = (float(row[37]), float(row[38]))
        d.p20 = (float(row[39]), float(row[40])) 
        d.p21 = (float(row[41]), float(row[42]))
        d.p22 = (float(row[43]), float(row[44]))
        d.p23 = (float(row[45]), float(row[46]))
        d.p24 = (float(row[47]), float(row[48]))
        d.p25 = (float(row[49]), float(row[50]))
        d.p26 = (float(row[51]), float(row[52]))
        d.p27 = (float(row[53]), float(row[54]))
        d.p28 = (float(row[55]), float(row[56]))
        d.p29 = (float(row[57]), float(row[58]))
        d.p30 = (float(row[59]), float(row[60])) 
        d.p31 = (float(row[61]), float(row[62]))
        d.p32 = (float(row[63]), float(row[64]))
        d.p33 = (float(row[65]), float(row[66]))
        d.p34 = (float(row[67]), float(row[68]))
        d.p35 = (float(row[69]), float(row[70]))
        d.p36 = (float(row[71]), float(row[72]))
        d.p37 = (float(row[73]), float(row[74]))
        d.p38 = (float(row[75]), float(row[76]))
        d.p39 = (float(row[77]), float(row[78]))
        d.p40 = (float(row[79]), float(row[80])) 
        d.p41 = (float(row[81]), float(row[82]))
        d.p42 = (float(row[83]), float(row[84]))
        d.p43 = (float(row[85]), float(row[86]))
        d.p44 = (float(row[87]), float(row[88]))
        d.p45 = (float(row[89]), float(row[90]))
        d.p46 = (float(row[91]), float(row[92]))
        d.p47 = (float(row[93]), float(row[94]))
        d.p48 = (float(row[95]), float(row[96]))
        d.p49 = (float(row[97]), float(row[98]))
        d.p50 = (float(row[99]), float(row[100]))
        d.p51 = (float(row[101]), float(row[102]))
        d.p52 = (float(row[103]), float(row[104]))
        d.p53 = (float(row[105]), float(row[106]))
        d.p54 = (float(row[107]), float(row[108]))
        d.p55 = (float(row[109]), float(row[110]))
        d.p56 = (float(row[111]), float(row[112]))
        d.p57 = (float(row[113]), float(row[114]))
        d.p58 = (float(row[115]), float(row[116]))
        d.p59 = (float(row[117]), float(row[118]))
        d.p60 = (float(row[119]), float(row[120]))
        d.p61 = (float(row[121]), float(row[122]))
        d.p62 = (float(row[123]), float(row[124]))
        d.p63 = (float(row[125]), float(row[126]))
        d.p64 = (float(row[127]), float(row[128]))
        d.p65 = (float(row[129]), float(row[130]))
        d.p66 = (float(row[131]), float(row[132]))
        d.p67 = (float(row[133]), float(row[134]))
        d.p68 = (float(row[135]), float(row[136]))

  

        return d

    @staticmethod
    def DataRowFromAFW(anno, root=''): # Assume data comming from parsed anno-v7.mat file.
        name = str(anno[0][0])
        bbox = anno[1][0][0]
#        yaw, pitch, roll = anno[2][0][0][0]
        lm = anno[3][0][0]  # 6 landmarks

        if np.isnan(lm).any():
            return None  # Fail

        d = DataRow()
        d.path = os.path.join(root, name).replace("\\", "/")
        d.name = os.path.split(d.path)[-1]
        d.image = cv2.imread(d.path)


        d.leftEye = (float(lm[0][0]), float(lm[0][1]))
        d.rightEye = (float(lm[1][0]), float(lm[1][1]))
        d.middle = (float(lm[2][0]), float(lm[2][1]))
        d.leftMouth = (float(lm[3][0]), float(lm[3][1]))
        # skip point 4 middle mouth - We take 0 left eye, 1 right eye, 2 nose, 3 left mouth, 5 right mouth
        d.rightMouth = (float(lm[5][0]), float(lm[5][1]))


        return d

    @staticmethod
    def DataRowFromPrediction(p, path='', image=None):
        d = DataRow(path)        
        p = (p+0.5)*60.  # scale from -0.5..+0.5 to 0..40
        
        d.p1 = (p[0], p[1])
        d.p2 = (p[2], p[3])
        d.p3 = (p[4], p[5])
        d.p4 = (p[6], p[7])
        d.p5 = (p[8], p[9])

        return d

    def drawLandmarks(self, r=3, color=255, other=None, title=None):
        M = self.image
        if hasattr(self, 'prediction'):
            for x,y in self.prediction.reshape(-1,2):
                cv2.circle(M, (int(x), int(y)), r, (0,200,0), -1)            

        cv2.circle(M, (int(self.p1[0]), int(self.p1[1])), r, color, -1)
        cv2.circle(M, (int(self.p2[0]), int(self.p2[1])), r, color, -1)
        cv2.circle(M, (int(self.p3[0]), int(self.p3[1])), r, color, -1)
        cv2.circle(M, (int(self.p4[0]), int(self.p4[1])), r, color, -1)
        cv2.circle(M, (int(self.p5[0]), int(self.p5[1])), r, color, -1)
        cv2.circle(M, (int(self.p6[0]), int(self.p6[1])), r, color, -1)
        cv2.circle(M, (int(self.p7[0]), int(self.p7[1])), r, color, -1)
        cv2.circle(M, (int(self.p8[0]), int(self.p8[1])), r, color, -1)
        cv2.circle(M, (int(self.p9[0]), int(self.p9[1])), r, color, -1)
        cv2.circle(M, (int(self.p10[0]), int(self.p10[1])), r, color, -1)
        cv2.circle(M, (int(self.p11[0]), int(self.p11[1])), r, color, -1)
        cv2.circle(M, (int(self.p12[0]), int(self.p12[1])), r, color, -1)
        cv2.circle(M, (int(self.p13[0]), int(self.p13[1])), r, color, -1)
        cv2.circle(M, (int(self.p14[0]), int(self.p14[1])), r, color, -1)
        cv2.circle(M, (int(self.p15[0]), int(self.p15[1])), r, color, -1)
        cv2.circle(M, (int(self.p16[0]), int(self.p16[1])), r, color, -1)
        cv2.circle(M, (int(self.p17[0]), int(self.p17[1])), r, color, -1)
        cv2.circle(M, (int(self.p18[0]), int(self.p18[1])), r, color, -1)
        cv2.circle(M, (int(self.p19[0]), int(self.p19[1])), r, color, -1)
        if hasattr(self, 'fbbox'):
            #cv2.rectangle(M, self.fbbox.top_left(), self.fbbox.bottom_right(), color)
#            cv2.rectangle(M, self.fbbox.left_top(), self.fbbox.right_bottom(), color)
            cv2.rectangle(M, (int(self.fbbox.left), int(self.fbbox.top)), (int(self.fbbox.right), int(self.fbbox.bottom)), color, 2)


            det_bbox = self.fbbox
            det_box = self.fbbox
            #height = (det_box.bottom - det_box.top)/1.35
            #det_bbox.top = det_box.top + height * 0.1
            #det_bbox.bottom = det_box.bottom - height*0.25
            #weight = (det_box.right - det_box.left)/1.3
            #det_bbox.left = det_box.left + weight*0.15
            #det_bbox.right = det_box.right - weight*0.15
            height = (det_box.bottom - det_box.top)/1.2
            det_bbox.top = det_box.top + height * 0.1
            det_bbox.bottom = det_box.bottom - height*0.1
            weight = (det_box.right - det_box.left)/1.2
            det_bbox.left = det_box.left + weight*0.1
            det_bbox.right = det_box.right - weight*0.1
            
            cv2.rectangle(M, (int(det_bbox.left), int(det_bbox.top)), (int(det_bbox.right), int(det_bbox.bottom)), (0,200,0), 2)
        return M

    def show(self, r=3, color=255, other=None, title=None):
        M = self.drawLandmarks(r, color, other, title)
        if title is None:
            title = self.name
        # my debug
        #cv2.imshow(title, M)

        return M
        
    def makeInt(self):
        self.p1 = (int(self.p1[0]), int(self.p1[1]))
        self.p2 = (int(self.p2[0]), int(self.p2[1]))
        self.p3 = (int(self.p3[0]), int(self.p3[1]))
        self.p4 = (int(self.p4[0]), int(self.p4[1]))
        self.p5 = (int(self.p5[0]), int(self.p5[1]))
        self.p6 = (int(self.p6[0]), int(self.p6[1]))
        self.p7 = (int(self.p7[0]), int(self.p7[1]))
        self.p8 = (int(self.p8[0]), int(self.p8[1]))
        self.p9 = (int(self.p9[0]), int(self.p9[1]))
        self.p10 = (int(self.p10[0]), int(self.p10[1]))
        self.p11 = (int(self.p11[0]), int(self.p11[1]))
        self.p12 = (int(self.p12[0]), int(self.p12[1]))
        self.p13 = (int(self.p13[0]), int(self.p13[1]))
        self.p14 = (int(self.p14[0]), int(self.p14[1]))
        self.p15 = (int(self.p15[0]), int(self.p15[1]))
        self.p16 = (int(self.p16[0]), int(self.p16[1]))
        self.p17 = (int(self.p17[0]), int(self.p17[1]))
        self.p18 = (int(self.p18[0]), int(self.p18[1]))
        self.p19 = (int(self.p19[0]), int(self.p19[1]))
        self.p20 = (int(self.p20[0]), int(self.p20[1]))
        self.p21 = (int(self.p21[0]), int(self.p21[1]))
        self.p22 = (int(self.p22[0]), int(self.p22[1]))
        self.p23 = (int(self.p23[0]), int(self.p23[1]))
        self.p24 = (int(self.p24[0]), int(self.p24[1]))
        self.p25 = (int(self.p25[0]), int(self.p25[1]))
        self.p26 = (int(self.p26[0]), int(self.p26[1]))
        self.p27 = (int(self.p27[0]), int(self.p27[1]))
        self.p28 = (int(self.p28[0]), int(self.p28[1]))
        self.p29 = (int(self.p29[0]), int(self.p29[1]))
        self.p30 = (int(self.p30[0]), int(self.p30[1]))
        self.p31 = (int(self.p31[0]), int(self.p31[1]))
        self.p32 = (int(self.p32[0]), int(self.p32[1]))
        self.p33 = (int(self.p33[0]), int(self.p33[1]))
        self.p34 = (int(self.p34[0]), int(self.p34[1]))
        self.p35 = (int(self.p35[0]), int(self.p35[1]))
        self.p36 = (int(self.p36[0]), int(self.p36[1]))
        self.p37 = (int(self.p37[0]), int(self.p37[1]))
        self.p38 = (int(self.p38[0]), int(self.p38[1]))
        self.p39 = (int(self.p39[0]), int(self.p39[1]))
        self.p40 = (int(self.p40[0]), int(self.p40[1]))
        self.p41 = (int(self.p41[0]), int(self.p41[1]))
        self.p42 = (int(self.p42[0]), int(self.p42[1]))
        self.p43 = (int(self.p43[0]), int(self.p43[1]))
        self.p44 = (int(self.p44[0]), int(self.p44[1]))
        self.p45 = (int(self.p45[0]), int(self.p45[1]))
        self.p46 = (int(self.p46[0]), int(self.p46[1]))
        self.p47 = (int(self.p47[0]), int(self.p47[1]))
        self.p48 = (int(self.p48[0]), int(self.p48[1]))
        self.p49 = (int(self.p49[0]), int(self.p49[1]))
        self.p50 = (int(self.p50[0]), int(self.p50[1]))
        self.p51 = (int(self.p51[0]), int(self.p51[1]))
        self.p52 = (int(self.p52[0]), int(self.p52[1]))
        self.p53 = (int(self.p53[0]), int(self.p53[1]))
        self.p54 = (int(self.p54[0]), int(self.p54[1]))
        self.p55 = (int(self.p55[0]), int(self.p55[1]))
        self.p56 = (int(self.p56[0]), int(self.p56[1]))
        self.p57 = (int(self.p57[0]), int(self.p57[1]))
        self.p58 = (int(self.p58[0]), int(self.p58[1]))
        self.p59 = (int(self.p59[0]), int(self.p59[1]))
        self.p60 = (int(self.p60[0]), int(self.p60[1]))
        self.p61 = (int(self.p61[0]), int(self.p61[1]))
        self.p62 = (int(self.p62[0]), int(self.p62[1]))
        self.p63 = (int(self.p63[0]), int(self.p63[1]))
        self.p64 = (int(self.p64[0]), int(self.p64[1]))
        self.p65 = (int(self.p65[0]), int(self.p65[1]))
        self.p66 = (int(self.p66[0]), int(self.p66[1]))
        self.p67 = (int(self.p67[0]), int(self.p67[1]))
        self.p68 = (int(self.p68[0]), int(self.p68[1]))

        return self        
         
    def copyCroppedByBBox(self,fbbox, siz=np.array([60.,60.])):
        """
        @ fbbox : BBox
        Returns a copy with cropped, scaled to size
        """        
        fbbox.makeInt() # assume BBox class
        if fbbox.width()<10 or fbbox.height()<10:
            print "Invalid bbox size:",fbbox
            return None
        #print "fbbox: ", fbbox
        faceOnly = self.image[fbbox.top : fbbox.bottom, fbbox.left:fbbox.right, :]
        scaled = DataRow() 
        scaled.image = cv2.resize(faceOnly, (int(siz[0]), int(siz[1])))                
        scaled.setLandmarks(self.landmarks())        
        """ @scaled: DataRow """
        scaled.offsetCropped(fbbox.left_top()) # offset the landmarks
        ry, rx = siz/faceOnly.shape[:2]
        scaled.scale(rx, ry)
        
        return scaled        
        
    def copyMirrored(self):
        '''
        Return a copy with mirrored data (and mirrored landmarks).
        '''
        import numpy
        _A=numpy.array
        ret = DataRow() 
        ret.image=cv2.flip(self.image.copy(),1)
        # Now we mirror the landmarks and swap left and right
        width = ret.image.shape[0] 
        ret.p1 = _A([width-self.p17[0], self.p17[1]]) # Toggle left\right eyes position and mirror x axis only
        ret.p2 = _A([width-self.p16[0], self.p16[1]])
        ret.p3 = _A([width-self.p15[0], self.p15[1]])        
        ret.p4 = _A([width-self.p14[0], self.p14[1]]) # Toggle mouth positions and mirror x axis only
        ret.p5 = _A([width-self.p13[0], self.p13[1]])
        ret.p6 = _A([width-self.p12[0], self.p12[1]])
        ret.p7 = _A([width-self.p11[0], self.p11[1]])
        ret.p8 = _A([width-self.p10[0], self.p10[1]])        
        ret.p9 = _A([width-self.p9[0], self.p9[1]])
        ret.p10 = _A([width-self.p8[0], self.p8[1]])
        ret.p11 = _A([width-self.p7[0], self.p7[1]]) # Toggle left\right eyes position and mirror x axis only
        ret.p12 = _A([width-self.p6[0], self.p6[1]])
        ret.p13 = _A([width-self.p5[0], self.p5[1]])        
        ret.p14 = _A([width-self.p4[0], self.p4[1]]) # Toggle mouth positions and mirror x axis only
        ret.p15 = _A([width-self.p3[0], self.p3[1]])
        ret.p16 = _A([width-self.p2[0], self.p2[1]])
        ret.p17 = _A([width-self.p1[0], self.p1[1]])
        ret.p18 = _A([width-self.p27[0], self.p27[1]])        
        ret.p19 = _A([width-self.p26[0], self.p26[1]])
        ret.p20 = _A([width-self.p25[0], self.p25[1]])
        ret.p21 = _A([width-self.p24[0], self.p24[1]])
        ret.p22 = _A([width-self.p23[0], self.p23[1]])
        ret.p23 = _A([width-self.p22[0], self.p22[1]])        
        ret.p24 = _A([width-self.p21[0], self.p21[1]])
        ret.p25 = _A([width-self.p20[0], self.p20[1]])
        ret.p26 = _A([width-self.p19[0], self.p19[1]])
        ret.p27 = _A([width-self.p18[0], self.p18[1]])
        ret.p28 = _A([width-self.p28[0], self.p28[1]])        
        ret.p29 = _A([width-self.p29[0], self.p29[1]])
        ret.p30 = _A([width-self.p30[0], self.p30[1]])
        ret.p31 = _A([width-self.p31[0], self.p31[1]]) # Toggle left\right eyes position and mirror x axis only
        ret.p32 = _A([width-self.p36[0], self.p36[1]])
        ret.p33 = _A([width-self.p35[0], self.p35[1]])        
        ret.p34 = _A([width-self.p34[0], self.p34[1]]) # Toggle mouth positions and mirror x axis only
        ret.p35 = _A([width-self.p33[0], self.p33[1]])
        ret.p36 = _A([width-self.p32[0], self.p32[1]])
        ret.p37 = _A([width-self.p46[0], self.p46[1]])
        ret.p38 = _A([width-self.p45[0], self.p45[1]])        
        ret.p39 = _A([width-self.p44[0], self.p44[1]])
        ret.p40 = _A([width-self.p43[0], self.p43[1]])
        ret.p41 = _A([width-self.p48[0], self.p48[1]]) # Toggle left\right eyes position and mirror x axis only
        ret.p42 = _A([width-self.p47[0], self.p47[1]])
        ret.p43 = _A([width-self.p40[0], self.p40[1]])        
        ret.p44 = _A([width-self.p39[0], self.p39[1]]) # Toggle mouth positions and mirror x axis only
        ret.p45 = _A([width-self.p38[0], self.p38[1]])
        ret.p46 = _A([width-self.p37[0], self.p37[1]])
        ret.p47 = _A([width-self.p42[0], self.p42[1]])
        ret.p48 = _A([width-self.p41[0], self.p41[1]])        
        ret.p49 = _A([width-self.p55[0], self.p55[1]])
        ret.p50 = _A([width-self.p54[0], self.p54[1]])
        ret.p51 = _A([width-self.p53[0], self.p53[1]]) # Toggle left\right eyes position and mirror x axis only
        ret.p52 = _A([width-self.p52[0], self.p52[1]])
        ret.p53 = _A([width-self.p51[0], self.p51[1]])        
        ret.p54 = _A([width-self.p50[0], self.p50[1]]) # Toggle mouth positions and mirror x axis only
        ret.p55 = _A([width-self.p49[0], self.p49[1]])
        ret.p56 = _A([width-self.p60[0], self.p60[1]])
        ret.p57 = _A([width-self.p59[0], self.p59[1]])
        ret.p58 = _A([width-self.p58[0], self.p58[1]])        
        ret.p59 = _A([width-self.p57[0], self.p57[1]])
        ret.p60 = _A([width-self.p56[0], self.p56[1]])
        ret.p61 = _A([width-self.p65[0], self.p65[1]]) # Toggle left\right eyes position and mirror x axis only
        ret.p62 = _A([width-self.p64[0], self.p64[1]])
        ret.p63 = _A([width-self.p63[0], self.p63[1]])        
        ret.p64 = _A([width-self.p62[0], self.p62[1]]) # Toggle mouth positions and mirror x axis only
        ret.p65 = _A([width-self.p61[0], self.p61[1]])
        ret.p66 = _A([width-self.p68[0], self.p68[1]])
        ret.p67 = _A([width-self.p67[0], self.p67[1]])
        ret.p68 = _A([width-self.p66[0], self.p66[1]])        

        return ret

    @staticmethod
    def dummyDataRow():
        ''' Returns a dummy dataRow object to play with
        '''
        return DataRow('/Users/ishay/Dev/VanilaCNN/data/train/lfw_5590/Abbas_Kiarostami_0001.jpg',
                     leftEye=(106.75, 108.25),
                     rightEye=(143.75,108.75) ,
                     middle = (131.25, 127.25),
                     leftMouth = (106.25, 155.25),
                     rightMouth =(142.75,155.25)
                     )    
        
  
            
class Predictor:
    ROOT = getGitRepFolder() 
    
    def preprocess(self, resized, landmarks):
        #ret = resized.astype('f4')
        #ret -= self.mean
        #ret /= (1.e-6+ self.std)
        #return  ret, (landmarks/40.)-0.5
        grayImg = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype('f4')
        m, s = cv2.meanStdDev(grayImg)
        grayImg = (grayImg-m)/(1.e-6 + s)
        return  grayImg, landmarks/60.
    
    def predict(self, resized):
        """
        @resized: image 40,40 already pre processed 
        """         
        #self.net.blobs['data'].data[...] = cv2.split(resized)
        self.net.blobs['data'].data[...] = resized.reshape(1,1,60,60)
        prediction = self.net.forward()['Dense3'][0]
        return prediction
        
    def __init__(self, protoTXTPath, weightsPath):
        import caffe
        caffe.set_mode_cpu()
        self.net = caffe.Net(protoTXTPath, weightsPath, caffe.TEST)
        self.mean = cv2.imread(os.path.join(Predictor.ROOT, 'trainMean.png')).astype('float')
        self.mean = cv2.resize(self.mean, (60,60), interpolation=cv2.INTER_CUBIC)
        self.std  = cv2.imread(os.path.join(Predictor.ROOT,'trainSTD.png')).astype('float')
        self.std = cv2.resize(self.std, (60,60), interpolation=cv2.INTER_CUBIC)

    
