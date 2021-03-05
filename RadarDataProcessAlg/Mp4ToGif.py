# encoding=utf-8
import argparse
import os

import cv2
import imageio

# name of the video file to convert
input_path = os.path.abspath('./output.mp4')


# targetFormat must be .gif


def ToGif(input_path,
          target_format,
          num_frames=60,  # max frame number
          out_size=(1600, 797)):  # (640, 360), (854, 480), (1920, 1080)
    """
    转换成gif格式
    """
    output_path = os.path.splitext(input_path)[0] + target_format  # 'codeblog', 'mp4'
    print('converting ', input_path, ' to ', output_path)

    # -----
    reader = imageio.get_reader(input_path)
    fps = reader.get_meta_data()['fps']
    writer = imageio.get_writer(output_path, fps=fps)

    for i, frame in enumerate(reader):
        if i < num_frames:
            frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_CUBIC)
            writer.append_data(frame)
            # print(f'frame: {frame}')

    writer.close()
    # -----

    print("Converting done.")


class Video2GifConverter(object):
    def __init__(self, video_path, out_f_path):
        if not os.path.isfile(video_path):
            print('[Err]: invalid video file path.')
            return

        self.in_f_path = video_path
        self.out_f_path = out_f_path

    def convert(self):
        reader = imageio.get_reader(self.in_f_path)

        fps = reader.get_meta_data()['fps']
        writer = imageio.get_writer(self.out_f_path, fps=fps)
        
        cnt = 0
        for frame in reader:
            writer.append_data(frame)
            cnt += 1
        print('Total {:d} frames.'.format(cnt))

        writer.close()
        print('Converting done.')

# ToGif(input_path, '.gif')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video',
                        type=str,
                        default='./output.mp4',
                        help='Video path to be processed')
    parser.add_argument('--frames',
                        type=int,
                        default=60,
                        help='Number of frames to be processed.')

    opt = parser.parse_args()

    ToGif(opt.video, '.gif', opt.frames)
