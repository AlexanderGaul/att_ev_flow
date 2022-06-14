import os
from pathlib import Path

import cv2
import imageio
import numpy

from data_generation.generate_events import EventGenerator
from data_generation.warp_sequences import MultiImageWarp


class SequenceWriter :
    def __init__(self, T, im_fps, flow_fps,
                 event_file, flow_dir, im_dir, flow_ts_file, im_ts_file,
                 min_events_per_frame, max_events_per_frame,
                 flow_backward_dir=None, flow_backward_ts_file=None,
                 event_count_file=None) :
        self.T = T
        self.im_fps = im_fps
        self.flow_fps = flow_fps
        self.event_file = event_file
        self.im_dir = im_dir
        self.flow_dir = flow_dir
        self.flow_backward_dir = flow_backward_dir
        self.flow_ts_file = flow_ts_file
        self.flow_backward_ts_file = flow_backward_ts_file
        self.im_ts_file = im_ts_file

        self.event_count_file = event_count_file

        self.min_events_per_frame = min_events_per_frame
        self.max_events_per_frame = max_events_per_frame

        self.event_writer = EventGenerator()

        self.test_first_frame = True

    @staticmethod
    def write_flow(flow:np.ndarray, path) :
        h, w, _ = flow.shape
        flow_map = np.rint(flow * 128 + 2 ** 15)
        flow_map = flow_map.astype(np.uint16) #.transpose(1, 2, 0)
        flow_map = np.concatenate((flow_map, np.ones((h, w, 1), dtype=np.uint16)), axis=-1)

        imageio.imwrite(path, flow_map, format='PNG-FI')

    @staticmethod
    def estimate_events(im, flow) :
        im = (im.mean(axis=-1) / 255).astype(float)
        log_im = np.log(np.minimum(im + 1e-4, np.ones(im.shape)))
        g_x = np.zeros(log_im.shape)
        g_y = np.zeros(log_im.shape)
        g_x = cv.Scharr(im, cv.CV_64FC1, 1, 0) / 16
        g_y = cv.Scharr(im, cv.CV_64FC1, 0, 1) / 16
        #g_x = cv.Sobel(im, cv.CV_64FC1, 1, 0, ksize=1)
        #g_y = cv.Sobel(im, cv.CV_64FC1, 0, 1, ksize=1)
        #g_x[:, :-1] = im[:, 1:] - im[:, :-1]
        #g_y[:-1, :] = im[1:, :] - im[:-1, :]

        g = np.stack([g_x, g_y], axis=-1)
        flow_length = np.linalg.norm(flow, axis=-1, ord=2, keepdims=True)
        flow_norm = flow / flow_length
        g = (g * flow_norm).sum(axis=-1)
        p = np.abs(g) / 0.2
        return (p * np.linalg.norm(flow, axis=-1, ord=2)).sum()


    def write_sequence(self, seq:MultiImageWarp, dir:Path, write_video=False,
                       remove_superfluous_images=True) :
        # [] create folders
        # [] what happens if folder already exists

        if not os.path.exists(dir) : os.makedirs(dir)
        if not os.path.exists(dir / self.im_dir) : os.makedirs(dir / self.im_dir)
        if not os.path.exists(dir / self.flow_dir) : os.makedirs(dir / self.flow_dir)
        if self.flow_backward_dir is not None and not os.path.exists(dir / self.flow_backward_dir) :
            os.makedirs(dir / self.flow_backward_dir)
        if not os.path.exists(os.path.dirname(dir / self.event_file)) :
            os.makedirs(os.path.dirname(dir / self.event_file))
        if not os.path.exists(os.path.dirname(dir / self.flow_ts_file)) :
            os.makedirs(os.path.dirname(dir / self.flow_ts_file))
        if not os.path.exists(os.path.dirname(dir / self.im_ts_file)):
            os.makedirs(os.path.dirname(dir / self.im_ts_file))

        im_ts = np.linspace(0, self.T, int(self.T * self.im_fps + 1))
        np.savetxt(dir / self.im_ts_file, im_ts)

        flow_ts_begin = np.linspace(0, self.T, int(self.T * self.flow_fps + 1))
        flow_ts = np.stack([flow_ts_begin[:-1] * 1e6, flow_ts_begin[1:] * 1e6]).transpose().astype(int)



        for f in os.listdir(dir / self.im_dir):
            os.remove(os.path.join(dir / self.im_dir, f))
        for f in os.listdir(dir / self.flow_dir):
            os.remove(os.path.join(dir / self.flow_dir, f))

        """# Estimate event count
        im = seq.get_image(im_ts[0])
        fl = seq.get_flow(flow_ts_begin[0], flow_ts_begin[1])
        print("estimated event count: " + str(self.estimate_events(im, fl)))"""


        if self.test_first_frame :
            ims_per_flow_frame = self.im_fps / self.flow_fps
            ims_to_test = int(ims_per_flow_frame + 1)
            for i, t in enumerate(im_ts[:ims_to_test]):
                im = seq.get_image(t)
                imageio.imwrite(dir / self.im_dir / ("0" * (6 - len(str(i))) + str(i) + ".png"), (im * 255).astype(np.uint8))

            assert len(os.listdir(dir / self.im_dir)) == ims_to_test
            events_first_frame = self.event_writer.generate_events(dir / self.im_dir,
                                                                   dir / self.im_ts_file)
            idx_last = np.searchsorted((events_first_frame[:, 2] * 1e6).astype(int), flow_ts[0, 1])
            if (idx_last < self.min_events_per_frame or
                    idx_last > self.max_events_per_frame) :
                print("First frame rejected" + ", events: " + str(idx_last))
                return False


        acc = 0.
        # [] write images to disc
        for i, t in enumerate(im_ts) :
            im = seq.get_image(t, borderValue=(0, 255, 0))
            if (im == np.array([0, 255, 0]).reshape(1,1,-1)).all(axis=-1).any() :
                print("Rejected because of border value")
                return False
            imageio.imwrite(dir / self.im_dir / ("0" * (6 - len(str(i))) + str(i) + ".png"), (im * 255).astype(np.uint8))

        # [] create events
        events = self.event_writer.generate_events(dir / self.im_dir,
                                                   dir / self.im_ts_file)
        print("Events generated")


        event_frame_idxs = np.searchsorted((events[:, 2]*1e6).astype(int), (flow_ts_begin*1e6).astype(int))
        event_counts = (event_frame_idxs[1:] - event_frame_idxs[:-1])
        if self.event_count_file is not None :
            np.savetxt(dir / self.event_count_file, event_counts, fmt="%d")
        print("Event counts: " + str(event_counts))
        if (event_counts < self.min_events_per_frame).any() or (event_counts > self.max_events_per_frame).any() :
            print("Rejected whole sequence")
            return False

        self.event_writer.write_events(events, dir / self.event_file)
        print("Events written")



        #flows = []
        #flows_backward = []
        np.savetxt(dir / self.flow_ts_file, flow_ts, fmt="%d")
        if self.flow_backward_ts_file is not None :
            np.savetxt(dir / self.flow_backward_ts_file, np.flip(flow_ts, axis=1), fmt="%d")
        for i, t in enumerate(flow_ts_begin[:-1]):
            fl = seq.get_flow(flow_ts_begin[i], flow_ts_begin[i + 1])
            #flows.append(fl)
            SequenceWriter.write_flow(fl, dir / self.flow_dir / ("0" * (6 - len(str(i))) + str(i) + ".png"))
            if self.flow_backward_dir is not None :
                fl_back = seq.get_flow(flow_ts_begin[i + 1], flow_ts_begin[i])
                SequenceWriter.write_flow(fl_back, dir / self.flow_backward_dir / ("0" * (6 - len(str(i))) + str(i) + ".png"))
            #flows_backward.append(fl_back)


        if remove_superfluous_images :
            counter = 1. -  self.flow_fps / self.im_fps
            for i, t in enumerate(im_ts) :
                counter += self.flow_fps / self.im_fps
                if counter >= 1. :
                    counter -= 1.
                    continue
                os.remove(dir / self.im_dir / ("0" * (6 - len(str(i))) + str(i) + ".png"))

        if write_video :
            ims = sorted(os.listdir(dir / self.im_dir))
            vid_writer = cv.VideoWriter(str(dir / "vid.avi"), 0, self.im_fps,
                                        (seq.warps[0].res[1], seq.warps[0].res[0]))
            for im_name in ims :
                im = cv.imread(str(dir / self.im_dir / im_name))
                vid_writer.write(im)
            vid_writer.release()



        return True