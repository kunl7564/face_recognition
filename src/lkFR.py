# -*- encoding=utf-8 -*-

from arcsoft import CLibrary, ASVL_COLOR_FORMAT, ASVLOFFSCREEN, AFD_FSDKLibrary, AFR_FSDKLibrary, c_ubyte_p, FaceInfo
from arcsoft.utils import BufferInfo, ImageLoader, FileUtils
from arcsoft.AFD_FSDKLibrary import *
from arcsoft.AFR_FSDKLibrary import *
from arcsoft.ASVL_COLOR_FORMAT import *
from ctypes import *
import traceback
import time
import cv2
import cv2.cv as cv
import sys
import os
from PIL.ImageFont import ImageFont
import platform
import thread
import threading
from win32api import GetSystemMetrics


# APPID = c_char_p(b'4vZmni7LpV3BJixEqAu44NDgzTAjd1thw7ivfSh4T5mF')
# FD_SDKKEY = c_char_p(b'HDnQWLCyTBaC323Xzwt8FMJorEdp4SDfNv6fGydyoP6u')
# FR_SDKKEY = c_char_p(b'HDnQWLCyTBaC323Xzwt8FMJw1du19Yt2xCKq3tZVEvcw')
APPID = c_char_p(b'4vZmni7LpV3BJixEqAu44NE4UexE4RqRazGm9x9ZrGg6')
FD_SDKKEY = c_char_p(b'3cDbDf8cHoVnMtRvMqgabBHNgeaCsecXefCtYdf5c3gN')
FR_SDKKEY = c_char_p(b'3cDbDf8cHoVnMtRvMqgabBHsLFctHd3ZE7WGc7YQY2Qf')
FD_WORKBUF_SIZE = 20 * 1024 * 1024
FR_WORKBUF_SIZE = 40 * 1024 * 1024
MAX_FACE_NUM = 50
bUseYUVFile = False
bUseBGRToEngine = True
nameList = []

def doFaceDetection(hFDEngine, inputImg):
    faceInfo = []

    pFaceRes = POINTER(AFD_FSDK_FACERES)()
    ret = AFD_FSDK_StillImageFaceDetection(hFDEngine, byref(inputImg), byref(pFaceRes))
    if ret != 0:
        print(u'AFD_FSDK_StillImageFaceDetection 0x{0:x}'.format(ret))
        return faceInfo

    faceRes = pFaceRes.contents
    if faceRes.nFace > 0:
        for i in range(0, faceRes.nFace):
            rect = faceRes.rcFace[i]
            orient = faceRes.lfaceOrient[i]
            faceInfo.append(FaceInfo(rect.left, rect.top, rect.right, rect.bottom, orient))

    return faceInfo

def extractFRFeature(hFREngine, inputImg, faceInfo):

        faceinput = AFR_FSDK_FACEINPUT()
        faceinput.lOrient = faceInfo.orient
        faceinput.rcFace.left = faceInfo.left
        faceinput.rcFace.top = faceInfo.top
        faceinput.rcFace.right = faceInfo.right
        faceinput.rcFace.bottom = faceInfo.bottom

        faceFeature = AFR_FSDK_FACEMODEL()
        ret = AFR_FSDK_ExtractFRFeature(hFREngine, inputImg, faceinput, faceFeature)
#         print "lFeatureSize=", faceFeature.lFeatureSize
        if ret != 0:
            print(u'AFR_FSDK_ExtractFRFeature ret 0x{0:x}'.format(ret))
            return None

        try:
            return faceFeature.deepCopy()
        except Exception as e:
            traceback.print_exc()
            print(e.message)
            return None

def compareFaceSimilarity(hFDEngine, hFREngine, inputImgA, inputImgB):
        # Do Face Detect
        faceInfosA = doFaceDetection(hFDEngine, inputImgA)
        if len(faceInfosA) < 1:
            print(u'no face in Image A ')
            return 0.0
        faceInfosB = doFaceDetection(hFDEngine, inputImgB)
        if len(faceInfosB) < 1:
            print(u'no face in Image B ')
            return 0.0

        # Extract Face Feature
        start_time = time.clock()
        faceFeatureA = extractFRFeature(hFREngine, inputImgA, faceInfosA[0])
        end_time = time.clock()
        print('extract time: %s Seconds' % (end_time - start_time))
        if faceFeatureA == None:
            print(u'extract face feature in Image A faile')
            return 0.0
        faceFeatureB = extractFRFeature(hFREngine, inputImgB, faceInfosB[0])
        if faceFeatureB == None:
            print(u'extract face feature in Image B failed')
            faceFeatureA.freeUnmanaged()
            return 0.0
        # calc similarity between faceA and faceB
        fSimilScore = c_float(0.0)
        start_time = time.clock()
        ret = AFR_FSDK_FacePairMatching(hFREngine, faceFeatureA, faceFeatureB, byref(fSimilScore))
        faceFeatureList = [faceFeatureA, faceFeatureB]
        end_time = time.clock()
        print('matching time: %s Seconds' % (end_time - start_time))
        faceFeatureA.freeUnmanaged()
        faceFeatureB.freeUnmanaged()
        if ret != 0:
            print(u'AFR_FSDK_FacePairMatching failed:ret 0x{0:x}'.format(ret))
            return 0.0
        return fSimilScore

def file_extension(path): 
  return os.path.splitext(path)[1]
# 
# def getFeatureByMem(hFDEngine, hFREngine, img):
#         inputImg = loadImage(inputImgPath)
#         # Do Face Detect
#         faceInfosA = doFaceDetection(hFDEngine, inputImg)
#         if len(faceInfosA) < 1:
# #             print(u'no face in Image ', inputImgPath)
#             return None
# 
#         # Extract Face Feature
#         faceFeature = extractFRFeature(hFREngine, inputImg, faceInfosA[0])
#         if faceFeature == None:
#             print(u'extract face feature in Image A failed')
#             return None
#         return faceFeature

def getFeature(hFDEngine, hFREngine, inputImgPath):
        inputImg = loadImage(inputImgPath)
        # Do Face Detect
        start_time = time.clock()
        faceInfosA = doFaceDetection(hFDEngine, inputImg)
        if len(faceInfosA) < 1:
#             print(u'no face in Image ', inputImgPath)
            return None
        end_time = time.clock()
        print "FaceDetection=", end_time - start_time
        # Extract Face Feature
        faceFeature = extractFRFeature(hFREngine, inputImg, faceInfosA[0])
        if faceFeature == None:
            print(u'extract face feature in Image A failed')
            return None
        return faceFeature

def addFeature2List(hFDEngine, hFREngine, inputImgPath, list):
        inputImg = loadImage(inputImgPath)
        # Do Face Detect
        faceInfosA = doFaceDetection(hFDEngine, inputImg)
        if len(faceInfosA) < 1:
            print(u'no face in Image A ')
            return False

        # Extract Face Feature
        faceFeature = extractFRFeature(hFREngine, inputImg, faceInfosA[0])
        if faceFeature == None:
            print(u'extract face feature in Image A failed')
            return False
        list.append(faceFeature)
        name = FileUtils.getFileNameAndExt(inputImgPath)[0].decode('gbk').encode("utf-8")
        nameList.append(name)
        return True

def initFeatureList(rootdir, hFDEngine, hFREngine, list):
    piclist = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(piclist)):
        path = os.path.join(rootdir, piclist[i])
        if os.path.isfile(path) and ".jpg" == file_extension(path):
            print path.decode('gbk').encode("utf-8")
            addFeature2List(hFDEngine, hFREngine, path, list)
    return True

def loadYUVImage(yuv_filePath, yuv_width, yuv_height, yuv_format):
    yuv_rawdata_size = 0

    inputImg = ASVLOFFSCREEN()
    inputImg.u32PixelArrayFormat = yuv_format
    inputImg.i32Width = yuv_width
    inputImg.i32Height = yuv_height
    if ASVL_COLOR_FORMAT.ASVL_PAF_I420 == inputImg.u32PixelArrayFormat:
        inputImg.pi32Pitch[0] = inputImg.i32Width
        inputImg.pi32Pitch[1] = inputImg.i32Width // 2
        inputImg.pi32Pitch[2] = inputImg.i32Width // 2
        yuv_rawdata_size = inputImg.i32Width * inputImg.i32Height * 3 // 2
    elif ASVL_COLOR_FORMAT.ASVL_PAF_NV12 == inputImg.u32PixelArrayFormat:
        inputImg.pi32Pitch[0] = inputImg.i32Width
        inputImg.pi32Pitch[1] = inputImg.i32Width
        yuv_rawdata_size = inputImg.i32Width * inputImg.i32Height * 3 // 2
    elif ASVL_COLOR_FORMAT.ASVL_PAF_NV21 == inputImg.u32PixelArrayFormat:
        inputImg.pi32Pitch[0] = inputImg.i32Width
        inputImg.pi32Pitch[1] = inputImg.i32Width
        yuv_rawdata_size = inputImg.i32Width * inputImg.i32Height * 3 // 2
    elif ASVL_COLOR_FORMAT.ASVL_PAF_YUYV == inputImg.u32PixelArrayFormat:
        inputImg.pi32Pitch[0] = inputImg.i32Width * 2
        yuv_rawdata_size = inputImg.i32Width * inputImg.i32Height * 2
    elif ASVL_COLOR_FORMAT.ASVL_PAF_RGB24_B8G8R8 == inputImg.u32PixelArrayFormat:
        inputImg.pi32Pitch[0] = inputImg.i32Width * 3
        yuv_rawdata_size = inputImg.i32Width * inputImg.i32Height * 3
    else:
        print(u'unsupported  yuv format')
        exit(0)

    # load YUV Image Data from File
    f = None
    try:
        f = open(yuv_filePath, u'rb')
        imagedata = f.read(yuv_rawdata_size)
    except Exception as e:
        traceback.print_exc()
        print(e.message)
        exit(0)
    finally:
        if f is not None:
            f.close()

    if ASVL_COLOR_FORMAT.ASVL_PAF_I420 == inputImg.u32PixelArrayFormat:
        inputImg.ppu8Plane[0] = cast(imagedata, c_ubyte_p)
        inputImg.ppu8Plane[1] = cast(addressof(inputImg.ppu8Plane[0].contents) + (inputImg.pi32Pitch[0] * inputImg.i32Height), c_ubyte_p)
        inputImg.ppu8Plane[2] = cast(addressof(inputImg.ppu8Plane[1].contents) + (inputImg.pi32Pitch[1] * inputImg.i32Height // 2), c_ubyte_p)
        inputImg.ppu8Plane[3] = cast(0, c_ubyte_p)
    elif ASVL_COLOR_FORMAT.ASVL_PAF_NV12 == inputImg.u32PixelArrayFormat:
        inputImg.ppu8Plane[0] = cast(imagedata, c_ubyte_p)
        inputImg.ppu8Plane[1] = cast(addressof(inputImg.ppu8Plane[0].contents) + (inputImg.pi32Pitch[0] * inputImg.i32Height), c_ubyte_p)
        inputImg.ppu8Plane[2] = cast(0, c_ubyte_p)
        inputImg.ppu8Plane[3] = cast(0, c_ubyte_p)
    elif ASVL_COLOR_FORMAT.ASVL_PAF_NV21 == inputImg.u32PixelArrayFormat:
        inputImg.ppu8Plane[0] = cast(imagedata, c_ubyte_p)
        inputImg.ppu8Plane[1] = cast(addressof(inputImg.ppu8Plane[0].contents) + (inputImg.pi32Pitch[0] * inputImg.i32Height), c_ubyte_p)
        inputImg.ppu8Plane[2] = cast(0, c_ubyte_p)
        inputImg.ppu8Plane[3] = cast(0, c_ubyte_p)
    elif ASVL_COLOR_FORMAT.ASVL_PAF_YUYV == inputImg.u32PixelArrayFormat:
        inputImg.ppu8Plane[0] = cast(imagedata, c_ubyte_p)
        inputImg.ppu8Plane[1] = cast(0, c_ubyte_p)
        inputImg.ppu8Plane[2] = cast(0, c_ubyte_p)
        inputImg.ppu8Plane[3] = cast(0, c_ubyte_p)
    elif ASVL_COLOR_FORMAT.ASVL_PAF_RGB24_B8G8R8 == inputImg.u32PixelArrayFormat:
        inputImg.ppu8Plane[0] = cast(imagedata, c_ubyte_p)
        inputImg.ppu8Plane[1] = cast(0, c_ubyte_p)
        inputImg.ppu8Plane[2] = cast(0, c_ubyte_p)
        inputImg.ppu8Plane[3] = cast(0, c_ubyte_p)
    else:
        print(u'unsupported yuv format')
        exit(0)

    inputImg.gc_ppu8Plane0 = imagedata
    return inputImg

def loadImage(filePath):

    inputImg = ASVLOFFSCREEN()
    if bUseBGRToEngine:
        bufferInfo = ImageLoader.getBGRFromFile(filePath)
        inputImg.u32PixelArrayFormat = ASVL_COLOR_FORMAT.ASVL_PAF_RGB24_B8G8R8
        inputImg.i32Width = bufferInfo.width
        inputImg.i32Height = bufferInfo.height
        inputImg.pi32Pitch[0] = bufferInfo.width * 3
        inputImg.ppu8Plane[0] = cast(bufferInfo.buffer, c_ubyte_p)
        inputImg.ppu8Plane[1] = cast(0, c_ubyte_p)
        inputImg.ppu8Plane[2] = cast(0, c_ubyte_p)
        inputImg.ppu8Plane[3] = cast(0, c_ubyte_p)
    else:
        bufferInfo = ImageLoader.getI420FromFile(filePath)
        inputImg.u32PixelArrayFormat = ASVL_COLOR_FORMAT.ASVL_PAF_I420
        inputImg.i32Width = bufferInfo.width
        inputImg.i32Height = bufferInfo.height
        inputImg.pi32Pitch[0] = inputImg.i32Width
        inputImg.pi32Pitch[1] = inputImg.i32Width // 2
        inputImg.pi32Pitch[2] = inputImg.i32Width // 2
        inputImg.ppu8Plane[0] = cast(bufferInfo.buffer, c_ubyte_p)
        inputImg.ppu8Plane[1] = cast(addressof(inputImg.ppu8Plane[0].contents) + (inputImg.pi32Pitch[0] * inputImg.i32Height), c_ubyte_p)
        inputImg.ppu8Plane[2] = cast(addressof(inputImg.ppu8Plane[1].contents) + (inputImg.pi32Pitch[1] * inputImg.i32Height // 2), c_ubyte_p)
        inputImg.ppu8Plane[3] = cast(0, c_ubyte_p)
    inputImg.gc_ppu8Plane0 = bufferInfo.buffer

    return inputImg

matchedName = ""
mOnReadingImg = False
mRun = True
lastUpdateTime = time.time()
class FaceRecognitionThread(threading.Thread):
    def __init__(self, name, hFDEngine, hFREngine, list):
        threading.Thread.__init__(self)
        self.name = name
        self.hFDEngine = hFDEngine
        self.hFREngine = hFREngine
        self.list = list
    def run(self):
        print "Starting " + self.name
        global mOnReadingImg
        global matchedName
        global condition
        while mRun:
            condition.acquire()
            condition.wait()
            condition.release()
            start_time = time.clock()
            tmpName = ""
            mOnReadingImg = True
            # 释放锁
            faceFeatureB = getFeature(hFDEngine, hFREngine, "lk.jpg")
            mOnReadingImg = False
            for i in range(0, len(list)):
                fSimilScore = c_float(0.0)
                feature = list[i]
                AFR_FSDK_FacePairMatching(hFREngine, feature, faceFeatureB, byref(fSimilScore))
                if fSimilScore.value > 0.5:
                    print nameList[i], "matched"
                    print(u'similarity is {0}'.format(fSimilScore))
                    tmpName = nameList[i]
            if tmpName != matchedName and time.time() - lastUpdateTime > 0.1:  # avoid to refresh too frequent
                matchedName = tmpName
            if faceFeatureB != None:
                faceFeatureB.freeUnmanaged();
            end_time = time.clock();
#             print "timecost=", (end_time - start_time);
 

threadLock = threading.Lock();
condition = threading.Condition()
def handleRealtimeVideo(window_name, camera_idx, hFDEngine, hFREngine, list):
    frThread = FaceRecognitionThread(window_name, hFDEngine, hFREngine, list)
    frThread.start()
    cv2.namedWindow(window_name, cv.CV_WINDOW_NORMAL)
    
    if platform.system() == u'Windows':
        screenWidth = GetSystemMetrics(0)
        screenHeight = GetSystemMetrics(1)
        boxSize = min(screenWidth, screenHeight)
    # 视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
    cap = cv2.VideoCapture(camera_idx)
 
    global mOnReadingImg
    global condition
    global matchedName
    while cap.isOpened():

        ok, frame = cap.read()  # 读取一帧数据
        if not ok:            
            break                
        # 显示图像并等待10毫秒按键输入，输入‘q’退出程序
        ret, arr = cv2.imencode('.jpg', frame)
        a = arr.tostring()
        if not mOnReadingImg:
            fp = open('lk.jpg', 'wb')
            fp.write(a)
            fp.close()
            condition.acquire()
            condition.notifyAll()
            condition.release()
        cv2.putText(frame, matchedName, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (0, 255, 255), thickness=2)

        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):            
            break
        if platform.system() == u'Windows':
#             resizedFrame = cv2.resize(frame, (boxSize, boxSize))
#             cv2.imshow(window_name, resizedFrame)
            cv2.imshow(window_name, frame)
        else:
            cv2.imshow(window_name, frame)
    
    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == u'__main__':
    print(u'#####################################################')
    ddir = sys.path[0]
    print "workdir=", os.getcwd(), "\nfiledir=", ddir
    os.chdir(ddir)
    start_time = time.clock()

    # init Engine
    pFDWorkMem = CLibrary.malloc(c_size_t(FD_WORKBUF_SIZE))
    pFRWorkMem = CLibrary.malloc(c_size_t(FR_WORKBUF_SIZE))

    hFDEngine = c_void_p()
    ret = AFD_FSDK_InitialFaceEngine(APPID, FD_SDKKEY, pFDWorkMem, c_int32(FD_WORKBUF_SIZE), byref(hFDEngine), AFD_FSDK_OPF_0_HIGHER_EXT, 32, MAX_FACE_NUM)
    if ret != 0:
        CLibrary.free(pFDWorkMem)
        print(u'AFD_FSDK_InitialFaceEngine ret 0x{:x}'.format(ret))
        exit(0)

    # print FDEngine version
    versionFD = AFD_FSDK_GetVersion(hFDEngine)
    print(u'{} {} {} {}'.format(versionFD.contents.lCodebase, versionFD.contents.lMajor, versionFD.contents.lMinor, versionFD.contents.lBuild))
    print(c_char_p(versionFD.contents.Version).value.decode(u'utf-8'))
    print(c_char_p(versionFD.contents.BuildDate).value.decode(u'utf-8'))
    print(c_char_p(versionFD.contents.CopyRight).value.decode(u'utf-8'))

    hFREngine = c_void_p()
    ret = AFR_FSDK_InitialEngine(APPID, FR_SDKKEY, pFRWorkMem, c_int32(FR_WORKBUF_SIZE), byref(hFREngine))
    if ret != 0:
        AFD_FSDKLibrary.AFD_FSDK_UninitialFaceEngine(hFDEngine)
        CLibrary.free(pFDWorkMem)
        CLibrary.free(pFRWorkMem)
        print(u'AFR_FSDK_InitialEngine ret 0x{:x}'.format(ret))
        System.exit(0)

    # print FREngine version
    versionFR = AFR_FSDK_GetVersion(hFREngine)
    print(u'{} {} {} {}'.format(versionFR.contents.lCodebase, versionFR.contents.lMajor, versionFR.contents.lMinor, versionFR.contents.lBuild))
    print(c_char_p(versionFR.contents.Version).value.decode(u'utf-8'))
    print(c_char_p(versionFR.contents.BuildDate).value.decode(u'utf-8'))
    print(c_char_p(versionFR.contents.CopyRight).value.decode(u'utf-8'))

#     # load Image Data
#     if bUseYUVFile:
#         filePathA = u'001_640x480_I420.YUV'
#         yuv_widthA = 640
#         yuv_heightA = 480
#         yuv_formatA = ASVL_COLOR_FORMAT.ASVL_PAF_I420
# 
#         filePathB = u'003_640x480_I420.YUV'
#         yuv_widthB = 640
#         yuv_heightB = 480
#         yuv_formatB = ASVL_COLOR_FORMAT.ASVL_PAF_I420
# 
#         inputImgA = loadYUVImage(filePathA, yuv_widthA, yuv_heightA, yuv_formatA)
#         inputImgB = loadYUVImage(filePathB, yuv_widthB, yuv_heightB, yuv_formatB)
#     else:
#         filePathA = u'face\liyanhong1.jpg'
#         filePathB = u'face\liyanhong2.jpg'
# 
#         inputImgA = loadImage(filePathA)
#         inputImgB = loadImage(filePathB)
# 
#     print(u'similarity between faceA and faceB is {0}'.format(compareFaceSimilarity(hFDEngine, hFREngine, inputImgA, inputImgB)))
    list = []
    initFeatureList("face", hFDEngine, hFREngine, list)
    handleRealtimeVideo("Tpson FaceRecognition", 0, hFDEngine, hFREngine, list)
    
    end_time = time.clock()
    print('total time: %s Seconds' % (end_time - start_time))

    # release Engine
    AFD_FSDK_UninitialFaceEngine(hFDEngine)
    AFR_FSDK_UninitialEngine(hFREngine)

    CLibrary.free(pFDWorkMem)
    CLibrary.free(pFRWorkMem)

    print(u'#####################################################')
