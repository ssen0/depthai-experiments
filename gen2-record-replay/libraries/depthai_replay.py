from pathlib import Path
import os
import cv2
import types
import depthai as dai

class Replay:
    disabled_streams = []
    stream_types = ['color', 'left', 'right', 'depth']

    def __init__(self, path):
        self.path = Path(path).resolve().absolute()

        self.frameSent = dict() # Last frame sent to the device
        self.frames = dict() # Frames read from Readers

        file_types = ['color', 'left', 'right', 'disparity', 'depth']
        extensions = ['mjpeg', 'avi', 'mp4', 'h265', 'h264', 'bag']

        self.readers = dict()
        for file in os.listdir(path):
            if not '.' in file: continue # Folder
            name, extension = file.split('.')
            if name in file_types and extension in extensions:
                if extension == 'bag':
                    from .video_readers.rosbag_reader import RosbagReader
                    self.readers[name] = RosbagReader(str(self.path / file))
                else:
                    from .video_readers.videocap_reader import VideoCapReader
                    self.readers[name] = VideoCapReader(str(self.path / file))

        if len(self.readers) == 0:
            raise RuntimeError("There are no recordings in the folder specified.")

        # Load calibration data from the recording folder
        self.calibData = dai.CalibrationHandler(str(self.path / "calib.json"))

        self.color_size = None
        # By default crop image as needed to keep the aspect ratio
        self.keep_ar = True

    # Resize color frames prior to sending them to the device
    def set_resize_color(self, size):
        self.color_size = size
    def keep_aspect_ratio(self, keep_aspect_ratio):
        self.keep_ar = keep_aspect_ratio

    def disable_stream(self, stream_name, disable_reading = False):
        if stream_name not in self.readers:
            print(f"There's no stream '{stream_name}' available!")
            return
        if disable_reading:
            self.readers[stream_name].close()
            # Remove the stream from the dict
            self.readers.pop(stream_name, None)

        self.disabled_streams.append(stream_name)

    def resize_color(self, frame):
        if self.color_size is None:
            # No resizing needed
            return frame

        if not self.keep_ar:
            # No need to keep aspect ratio, image will be squished
            return cv2.resize(frame, self.color_size)

        h = frame.shape[0]
        w = frame.shape[1]
        desired_ratio = self.color_size[0] / self.color_size[1]
        current_ratio = w / h

        # Crop width/heigth to match the aspect ratio needed by the NN
        if desired_ratio < current_ratio: # Crop width
            # Use full height, crop width
            new_w = (desired_ratio/current_ratio) * w
            crop = int((w - new_w) / 2)
            preview = frame[:, crop:w-crop]
        else: # Crop height
            # Use full width, crop height
            new_h = (current_ratio/desired_ratio) * h
            crop = int((h - new_h) / 2)
            preview = frame[crop:h-crop,:]

        return cv2.resize(preview, self.color_size)

    def init_pipeline(self):
        pipeline = dai.Pipeline()
        pipeline.setCalibrationData(self.calibData)
        nodes = types.SimpleNamespace()

        def createXIn(p: dai.Pipeline, name: str):
            xin = p.create(dai.node.XLinkIn)
            xin.setMaxDataSize(self.get_max_size(name))
            xin.setStreamName(name + '_in')
            return xin

        for name in self.readers:
            if name not in self.disabled_streams:
                setattr(nodes, name, createXIn(pipeline, name))
        print(nodes)

        if hasattr(nodes, 'left') and hasattr(nodes, 'right'): # Create StereoDepth node
            nodes.stereo = pipeline.create(dai.node.StereoDepth)
            nodes.stereo.setInputResolution(self.readers['left'].getShape())

            nodes.left.out.link(nodes.stereo.left)
            nodes.right.out.link(nodes.stereo.right)

        return pipeline, nodes

    def create_queues(self, device):
        self.queues = dict()
        for name in self.readers:
            if name in self.stream_types and name not in self.disabled_streams:
                self.queues[name+'_in'] = device.getInputQueue(name+'_in')

    def to_planar(self, arr, shape = None):
        if shape is not None: arr = cv2.resize(arr, shape)
        return arr.transpose(2, 0, 1).flatten()

    def read_frames(self):
        self.frames = dict()
        for name in self.readers:
            self.frames[name] = self.readers[name].read() # Read the frame
            if self.frames[name] is False:
                return True # No more frames!

    def send_frames(self):
        if self.read_frames():
            return False # end of recording
        for name in self.frames:
            if name in ["left", "right", "disparity"] and len(self.frames[name].shape) == 3:
                self.frames[name] = self.frames[name][:,:,0] # All 3 planes are the same

            # Don't send these frames to the OAK camera
            if name in self.disabled_streams: continue

            self.send_frame(self.frames[name], name)

        return True

    def get_max_size(self, name):
        size = self.readers[name].getShape()
        bytes_per_pixel = 1
        if name == 'color': bytes_per_pixel = 3
        elif name == 'depth': bytes_per_pixel = 2 # 16bit
        return size[0] * size[1] * bytes_per_pixel

    def send_frame(self, frame, name):
        q_name = name + '_in'
        if q_name in self.queues:
            if name == 'color':
                # Resize/crop color frame as specified by the user
                frame = self.resize_color(frame)
                self.send_color(self.queues[q_name], frame)
            elif name == 'left':
                self.send_mono(self.queues[q_name], frame, False)
            elif name == 'right':
                self.send_mono(self.queues[q_name], frame, True)
            elif name == 'depth':
                self.send_depth(self.queues[q_name], frame)

            # Save the sent frame
            self.frameSent[name] = frame

    def send_mono(self, q, img, right):
        h, w = img.shape
        frame = dai.ImgFrame()
        frame.setData(img)
        frame.setType(dai.RawImgFrame.Type.RAW8)
        frame.setWidth(w)
        frame.setHeight(h)
        frame.setInstanceNum((2 if right else 1))
        q.send(frame)

    def send_color(self, q, img):
        h, w, c = img.shape
        frame = dai.ImgFrame()
        frame.setType(dai.RawImgFrame.Type.BGR888p)
        frame.setData(self.to_planar(img))
        frame.setWidth(w)
        frame.setHeight(h)
        frame.setInstanceNum(0)
        q.send(frame)

    def send_depth(self, q, depth):
        # TODO refactor saving depth. Reading will be from ROS bags.

        # print("depth size", type(depth))
        # depth_frame = np.array(depth).astype(np.uint8).view(np.uint16).reshape((400, 640))
        # depthFrameColor = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        # depthFrameColor = cv2.equalizeHist(depthFrameColor)
        # depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
        # cv2.imshow("depth", depthFrameColor)
        frame = dai.ImgFrame()
        frame.setType(dai.RawImgFrame.Type.RAW16)
        frame.setData(depth)
        frame.setWidth(640)
        frame.setHeight(400)
        frame.setInstanceNum(0)
        q.send(frame)

    def close(self):
        for name in self.readers:
            self.readers[name].close()