
import cv2
import os


class loader:  # for inference
    def __init__(self, path):
        self.frame = 0
        assert os.path.exists(path), path+' not exists!'
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    def __iter__(self):
        self.count = 0
        return self

    def next(self):
        ret_val, img0 = self.cap.read()
        if not ret_val:
            self.count += 1
            self.cap.release()
        self.frame += 1
        # print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nf, self.frame, self.nframes, path), end='')

        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return [img0, self.cap]
