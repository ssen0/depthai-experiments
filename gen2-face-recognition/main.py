# coding=utf-8
import os
from pathlib import Path
import argparse
import blobconverter
import cv2
import depthai as dai
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the person for database saving")

args = parser.parse_args()

databases = "databases"
if not os.path.exists(databases):
    os.mkdir(databases)


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

    def new_recognition(self, frame, coords, results):
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
        self.putText(frame, f"name:{name[1]}", (coords[0], coords[1] - 35))
        self.putText(frame, f"conf:{name[0] * 100:.2f}%", (coords[0], coords[1] - 10))

        if name[1] == "UNKNOWN":
            self.create_db(results)

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

def create_pipeline():
    print("Creating pipeline...")
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_2)
    openvino_version = '2021.2'

    print("Creating Color Camera...")
    cam = pipeline.create(dai.node.ColorCamera)
    # For ImageManip rotate you need input frame of multiple of 16
    cam.setPreviewSize(1072, 1072)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)

    # ImageManip that will crop the frame before sending it to the Face detection NN node
    face_det_manip = pipeline.create(dai.node.ImageManip)
    face_det_manip.initialConfig.setResize(300, 300)
    face_det_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)

    # NeuralNetwork
    print("Creating Face Detection Neural Network...")
    face_det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    face_det_nn.setConfidenceThreshold(0.5)
    face_det_nn.setBlobPath(str(blobconverter.from_zoo(
        name="face-detection-retail-0004",
        shaves=6,
        version=openvino_version
    )))
    # Link Face ImageManip -> Face detection NN node
    face_det_manip.out.link(face_det_nn.input)

    # Script node will take the output from the face detection NN as an input and set ImageManipConfig
    # to the 'age_gender_manip' to crop the initial frame
    script = pipeline.create(dai.node.Script)

    face_det_nn.out.link(script.inputs['face_det_in'])
    # We are only interested in timestamp, so we can sync depth frames with NN output
    face_det_nn.passthrough.link(script.inputs['face_pass'])

    with open("script.py", "r") as f:
        script.setScript(f.read())

    # ImageManip as a workaround to have more frames in the pool.
    # cam.preview can only have 4 frames in the pool before it will
    # wait (freeze). Copying frames and setting ImageManip pool size to
    # higher number will fix this issue.
    copy_manip = pipeline.create(dai.node.ImageManip)
    cam.preview.link(copy_manip.inputImage)
    copy_manip.setNumFramesPool(16)
    copy_manip.setMaxOutputFrameSize(1072*1072*3)

    copy_manip.out.link(face_det_manip.inputImage)
    copy_manip.out.link(script.inputs['preview'])

    # Send face synced detections (for bounding boxes) and color frames to the host
    host_face_out = pipeline.create(dai.node.XLinkOut)
    host_face_out.setStreamName('frame')
    script.outputs['host_frame'].link(host_face_out.input)

    print("Creating Head pose estimation NN")
    headpose_manip = pipeline.create(dai.node.ImageManip)
    headpose_manip.initialConfig.setResize(60, 60)

    script.outputs['manip_cfg'].link(headpose_manip.inputConfig)
    script.outputs['manip_img'].link(headpose_manip.inputImage)

    headpose_nn = pipeline.create(dai.node.NeuralNetwork)
    headpose_nn.setBlobPath(str(blobconverter.from_zoo(
        name="head-pose-estimation-adas-0001",
        shaves=6,
        version=openvino_version
    )))
    headpose_manip.out.link(headpose_nn.input)

    headpose_nn.out.link(script.inputs['headpose_in'])
    headpose_nn.passthrough.link(script.inputs['headpose_pass'])

    print("Creating face recognition ImageManip/NN")

    face_rec_manip = pipeline.create(dai.node.ImageManip)
    face_rec_manip.initialConfig.setResize(112, 112)

    script.outputs['manip2_cfg'].link(face_rec_manip.inputConfig)
    script.outputs['manip2_img'].link(face_rec_manip.inputImage)

    face_rec_cfg_out = pipeline.create(dai.node.XLinkOut)
    face_rec_cfg_out.setStreamName('face_rec_cfg_out')
    script.outputs['manip2_cfg'].link(face_rec_cfg_out.input)

    # rotate_out = pipeline.create(dai.node.XLinkOut)
    # rotate_out.setStreamName('rotated_frame')
    # face_rec_manip.out.link(rotate_out.input)

    face_rec_nn = pipeline.create(dai.node.NeuralNetwork)
    # Removed from OMZ, so we can't use blobconverter for downloading, see here:
    # https://github.com/openvinotoolkit/open_model_zoo/issues/2448#issuecomment-851435301
    face_rec_nn.setBlobPath("models/face-recognition-mobilefacenet-arcface_2021.2_4shave.blob")
    face_rec_manip.out.link(face_rec_nn.input)

    arc_out = pipeline.create(dai.node.XLinkOut)
    arc_out.setStreamName('arc_out')
    face_rec_nn.out.link(arc_out.input)

    return pipeline


with dai.Device(create_pipeline()) as device:
    frameQ = device.getOutputQueue("frame", 4, False)
    # rotateQ = device.getOutputQueue("rotated_frame", 4, False)
    recCfgQ = device.getOutputQueue("face_rec_cfg_out", 4, False)
    arcQ = device.getOutputQueue("arc_out", 4, False)

    facerec = FaceRecognition(databases, args.name)
    frame = None

    while True:
        frameIn = frameQ.tryGet()
        if frameIn is not None:
            frame = frameIn.getCvFrame()

        face_in = recCfgQ.tryGet()
        if face_in is not None and frame is not None:
            print('New det')
            rr = face_in.getRaw().cropConfig.cropRotatedRect
            h, w, c = frame.shape
            center = (int(rr.center.x * w), int(rr.center.y * h))
            size = (int(rr.size.width * w), int(rr.size.height * h))
            rotatedRect = (center, size, rr.angle)
            points = np.int0(cv2.boxPoints(rotatedRect))
            cv2.drawContours(frame, [points], 0, (255, 0, 0), 3)
            # draw_detections(frame, face_in.detections)
            print('1')
            arcIn = arcQ.get()
            print('2')
            features = np.array(arcIn.getFirstLayerFp16())
            # facerec.new_recognition(frame, center, features)


        # rotateIn = rotateQ.tryGet()
        # if rotateIn is not None:
        #     cv2.imshow('rotate', rotateIn.getCvFrame())
        if frame is not None:
            cv2.imshow("color", cv2.resize(frame, (500,500)))

        if cv2.waitKey(1) == ord('q'):
            break
