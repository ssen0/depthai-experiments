# coding=utf-8
import os
import argparse
import blobconverter
import cv2
import depthai as dai
import numpy as np
import time
from MultiMsgSync import TwoStageHostSeqSync

import zbarlight
from PIL import Image
import threading
from multiprocessing import Queue
import time
import subprocess

import queue
import urllib.request
import json
import re

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the person for database saving")

args = parser.parse_args()

def frame_norm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

VIDEO_SIZE = (800, 800)
databases = "databases"
if not os.path.exists(databases):
    os.mkdir(databases)

class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA

    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 1.0, self.bg_color, 4, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 1.0, self.color, 2, self.line_type)

class FaceRecognition:
    def __init__(self, db_path, name) -> None:
        self.read_db(db_path)
        self.name = name
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
        self.printed = True

    def cosine_distance(self, a, b):
        if a.shape != b.shape:
            raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        return np.dot(a, b.T) / (a_norm * b_norm)

    def new_recognition(self, results):
        conf = []
        max_ = 0
        label_ = None
        for label in list(self.labels):
            for j in self.db_dic.get(label):
                conf_ = self.cosine_distance(j, results)
                if conf_ > max_:
                    max_ = conf_
                    label_ = label

        conf.append((max_, label_))
        name = conf[0] if conf[0][0] >= 0.5 else (1 - conf[0][0], "UNKNOWN")
        # self.putText(frame, f"name:{name[1]}", (coords[0], coords[1] - 35))
        # self.putText(frame, f"conf:{name[0] * 100:.2f}%", (coords[0], coords[1] - 10))

        if name[1] == "UNKNOWN":
            self.create_db(results)
        return name

    def read_db(self, databases_path):
        self.labels = []
        for file in os.listdir(databases_path):
            filename = os.path.splitext(file)
            if filename[1] == ".npz":
                self.labels.append(filename[0])

        self.db_dic = {}
        for label in self.labels:
            with np.load(f"{databases_path}/{label}.npz") as db:
                self.db_dic[label] = [db[j] for j in db.files]

    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 1, self.bg_color, 4, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 1, self.color, 1, self.line_type)

    def create_db(self, results):
        if self.name is None:
            if not self.printed:
                print("Wanted to create new DB for this face, but --name wasn't specified")
                self.printed = True
            return
        print('Saving face...')
        try:
            with np.load(f"{databases}/{self.name}.npz") as db:
                db_ = [db[j] for j in db.files][:]
        except Exception as e:
            db_ = []
        db_.append(np.array(results))
        np.savez_compressed(f"{databases}/{self.name}", *db_)
        self.adding_new = False

print("Creating pipeline...")
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_2)
openvino_version = '2021.2'

print("Creating Color Camera...")
cam = pipeline.create(dai.node.ColorCamera)
# For ImageManip rotate you need input frame of multiple of 16
cam.setPreviewSize(800, 800)
cam.setVideoSize(VIDEO_SIZE)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setInterleaved(False)
cam.setBoardSocket(dai.CameraBoardSocket.RGB)

host_face_out = pipeline.create(dai.node.XLinkOut)
host_face_out.setStreamName('color')
cam.video.link(host_face_out.input)

# ImageManip as a workaround to have more frames in the pool.
# cam.preview can only have 4 frames in the pool before it will
# wait (freeze). Copying frames and setting ImageManip pool size to
# higher number will fix this issue.
copy_manip = pipeline.create(dai.node.ImageManip)
cam.preview.link(copy_manip.inputImage)
copy_manip.setNumFramesPool(20)
copy_manip.setMaxOutputFrameSize(800*800*3)

# ImageManip that will crop the frame before sending it to the Face detection NN node
face_det_manip = pipeline.create(dai.node.ImageManip)
face_det_manip.initialConfig.setResize(300, 300)
copy_manip.out.link(face_det_manip.inputImage)

# NeuralNetwork
print("Creating Face Detection Neural Network...")
face_det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
face_det_nn.setConfidenceThreshold(0.5)
face_det_nn.setBlobPath(blobconverter.from_zoo(
    name="face-detection-retail-0004",
    shaves=6,
    version=openvino_version
))
# Link Face ImageManip -> Face detection NN node
face_det_manip.out.link(face_det_nn.input)

face_det_xout = pipeline.create(dai.node.XLinkOut)
face_det_xout.setStreamName("detection")
face_det_nn.out.link(face_det_xout.input)

# Script node will take the output from the face detection NN as an input and set ImageManipConfig
# to the 'age_gender_manip' to crop the initial frame
script = pipeline.create(dai.node.Script)
script.setProcessor(dai.ProcessorType.LEON_CSS)

face_det_nn.out.link(script.inputs['face_det_in'])
# We also interested in sequence number for syncing
face_det_nn.passthrough.link(script.inputs['face_pass'])

copy_manip.out.link(script.inputs['preview'])

with open("script.py", "r") as f:
    script.setScript(f.read())

print("Creating Head pose estimation NN")

headpose_manip = pipeline.create(dai.node.ImageManip)
headpose_manip.initialConfig.setResize(60, 60)
headpose_manip.setWaitForConfigInput(True)
script.outputs['manip_cfg'].link(headpose_manip.inputConfig)
script.outputs['manip_img'].link(headpose_manip.inputImage)

headpose_nn = pipeline.create(dai.node.NeuralNetwork)
headpose_nn.setBlobPath(blobconverter.from_zoo(
    name="head-pose-estimation-adas-0001",
    shaves=6,
    version=openvino_version
))
headpose_manip.out.link(headpose_nn.input)

headpose_nn.out.link(script.inputs['headpose_in'])
headpose_nn.passthrough.link(script.inputs['headpose_pass'])

print("Creating face recognition ImageManip/NN")

face_rec_manip = pipeline.create(dai.node.ImageManip)
face_rec_manip.initialConfig.setResize(112, 112)
face_rec_manip.setWaitForConfigInput(True)

script.outputs['manip2_cfg'].link(face_rec_manip.inputConfig)
script.outputs['manip2_img'].link(face_rec_manip.inputImage)

face_rec_nn = pipeline.create(dai.node.NeuralNetwork)
# Removed from OMZ, so we can't use blobconverter for downloading, see here:
# https://github.com/openvinotoolkit/open_model_zoo/issues/2448#issuecomment-851435301
face_rec_nn.setBlobPath("models/face-recognition-mobilefacenet-arcface_2021.2_4shave.blob")
face_rec_manip.out.link(face_rec_nn.input)

arc_xout = pipeline.create(dai.node.XLinkOut)
arc_xout.setStreamName('recognition')
face_rec_nn.out.link(arc_xout.input)

img_q = Queue(3)
qr_name = ""
rec_name = ""

fr_thread_flag = True
qr_thread_flag = True
rh_thread_flag = True

def face_rec(id_name):
    with dai.Device(pipeline) as device:
        facerec = FaceRecognition(databases, id_name)
        sync = TwoStageHostSeqSync()
        text = TextHelper()

        queues = {}
        # Create output queues
        for name in ["color", "detection", "recognition"]:
            queues[name] = device.getOutputQueue(name)

        global fr_thread_flag
        global rec_name

        while True:
            for name, q in queues.items():
                # Add all msgs (color frames, object detections and face recognitions) to the Sync class.
                if q.has():
                    sync.add_msg(q.get(), name)

            msgs = sync.get_msgs()
            if msgs is not None:
                frame = msgs["color"].getCvFrame()
                dets = msgs["detection"].detections

                for i, detection in enumerate(dets):
                    bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)

                    features = np.array(msgs["recognition"][i].getFirstLayerFp16())
                    conf, name = facerec.new_recognition(features)
                    text.putText(frame, f"{name} {(100*conf):.0f}%", (bbox[0] + 10,bbox[1] + 35))

                    if 100*conf > 85 and name != "UNKNOWN":
                        rec_name = name

                cv2.imshow("color", frame)

                img_q.put(frame)

                # snapshot = cv2.resize(frame, (200,200))
                # im_rgb = cv2.cvtColor(snapshot, cv2.COLOR_BGR2RGB)
                # PIL_data = Image.fromarray(im_rgb)
                # codes = zbarlight.scan_codes('qrcode',PIL_data)
                # print(f'QR codes: {codes}')

            if not fr_thread_flag:
                cv2.destroyAllWindows()
                break

            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                global qr_thread_flag
                qr_thread_flag = False
                break

def qrcode_reader():
    global qr_thread_flag

    while True:
        try:
            q = img_q.get(block=True, timeout=8)
            snapshot = cv2.resize(q, (200,200))
            im_rgb = cv2.cvtColor(snapshot, cv2.COLOR_BGR2RGB)
            PIL_data = Image.fromarray(im_rgb)
            codes = zbarlight.scan_codes('qrcode',PIL_data)

            if codes != None:
                print(f'QR codes: {codes}')
                global fr_thread_flag
                fr_thread_flag = False

                global qr_name
                qr_name = codes

                #queueを空にする
                while True:
                    img_q.get()
                    if img_q.empty():
                        print("empty")
                        break
                break
        except queue.Empty:
            print("empty")

        if not qr_thread_flag:
            break

def rec_http():
    global rh_thread_flag
    global rec_name
    while True:
        if rec_name != "":
            result = re.split('_', rec_name)
            id = result[0]
            nickname = result[1]

            # 送信先のURL
            url = 'http://192.168.63.174:3000/admin/reserves/' + id

            #body作成
            req_data = json.dumps({
                "nickname": nickname,
                "status": "認証済み", 
            })

            # header作成
            req_header = {'Content-Type': 'application/json',
                        'access-token': 'UgeZptuUv3M5PhtTwzrN6g',
                        'client': 'aX-AhIA70a0DBYhF6DUd6A',
                        'uid': 'admin@example.com',
                        'expiry': '1657515044'
                        }

            #リクエスト処理
            req = urllib.request.Request(url, data=req_data.encode(), method='PUT', headers=req_header)
            try:
                with urllib.request.urlopen(req) as response:
                    body = json.loads(response.read())
                    header = response.getheaders()
                    status = response.getcode()
                    print(response.read())
                    print(header)
                    print(status)
            except urllib.error.URLError as e:
                print(e.reason)

            time.sleep(10)
            rec_name = ""

        if not rh_thread_flag:
            break

fr_thread = threading.Thread(target=face_rec, args=(None,))
fr_thread.start()

qr_thread = threading.Thread(target=qrcode_reader)
qr_thread.start()

rh_thread = threading.Thread(target=rec_http)
rh_thread.start()

fr_thread.join()
qr_thread.join()

rh_thread_flag = False
rh_thread.join()

if qr_thread_flag:
    #qr_name = "12_文字"
    print("カメラに顔全体が収まるように近づいてください")
    subprocess.run(["python3", "register.py", "-name", qr_name[0]])

