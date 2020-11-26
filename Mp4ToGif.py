# encoding=utf-8
import imageio
import os
import cv2

# name of the video file to convert
input_path = os.path.abspath('f:/MVI_40855_track_fps12.mp4')

# targetFormat must be .gif


def ToGif(input_path,
          target_format,
          num_frames=56,         # max frame number
          out_size=(640, 360)):  # (640, 360), (854, 480), (1920, 1080)
    """
    转换成gif格式
    """
    output_path = os.path.splitext(
        input_path)[0] + target_format  # 'codeblog', 'mp4'
    print('converting ', input_path, ' to ', output_path)

    # -----
    reader = imageio.get_reader(input_path)
    fps = reader.get_meta_data()['fps']
    writer = imageio.get_writer(output_path, fps=fps)

    for i, frame in enumerate(reader):
        if i < num_frames:
            frame = cv2.resize(frame, out_size, interpolation=cv2.INTER_CUBIC)
            writer.append_data(frame)
            #print(f'frame: {frame}')

    writer.close()
    # -----

    print("Converting done.")


ToGif(input_path, '.gif')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video',
                        type=str,
                        default='f:/MVI_40855_track_fps12.mp4',
                        help='Video path to be processed')
    parser.add_argument('--frames',
                        type=int,
                        default=60,
                        help='Number of frames to be processed.')

    opt = parser.parse_args()
    
    ToGif(opt.video, opt.frames)
    
