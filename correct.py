import os
import shutil
import sys
import threading

import numpy as np
import cv2
import math
import skimage

from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from skimage import img_as_ubyte

import CustomLogger
from CustomLogger import video_proc_logger, audio_proc_logger

###############  underwater color correction #####################################

THRESHOLD_RATIO = 2000
MIN_AVG_RED = 60
MAX_HUE_SHIFT = 120
BLUE_MAGIC_VALUE = 1.2
# Extracts color correction from every N seconds
#if set 0 - every frame will be analyzed. if set value > 0 - will be analyzed every N second. if set -1 - will be analized only first frame
SAMPLE_SECONDS = 2



video_fps = 0.0

# temp folder config
temp_dir = 'Temp'
temp_dir_path = './' + temp_dir + '/'

def hue_shift_red(mat, h):
    # print('called hue_shift_red')
    U = math.cos(h * math.pi / 180)
    W = math.sin(h * math.pi / 180)

    r = (0.299 + 0.701 * U + 0.168 * W) * mat[..., 0]
    g = (0.587 - 0.587 * U + 0.330 * W) * mat[..., 1]
    b = (0.114 - 0.114 * U - 0.497 * W) * mat[..., 2]

    return np.dstack([r, g, b])


def normalizing_interval(array):
    # print('called normalizing_interval')
    high = 255
    low = 0
    max_dist = 0

    for i in range(1, len(array)):
        dist = array[i] - array[i - 1]
        if (dist > max_dist):
            max_dist = dist
            high = array[i]
            low = array[i - 1]

    return (low, high)


def apply_filter(mat, filt):
    # print('called apply_filter')
    r = mat[..., 0]
    g = mat[..., 1]
    b = mat[..., 2]

    r = r * filt[0] + g * filt[1] + b * filt[2] + filt[4] * 255
    g = g * filt[6] + filt[9] * 255
    b = b * filt[12] + filt[14] * 255

    filtered_mat = np.dstack([r, g, b])
    filtered_mat = np.clip(filtered_mat, 0, 255).astype(np.uint8)

    return filtered_mat


def get_filter_matrix(mat):
    # print('called get_filter_matrix')
    mat = cv2.resize(mat, (256, 256))

    # Get average values of RGB
    avg_mat = np.array(cv2.mean(mat)[:3], dtype=np.uint8)

    # Find hue shift so that average red reaches MIN_AVG_RED
    new_avg_r = avg_mat[0]
    hue_shift = 0
    while (new_avg_r < MIN_AVG_RED):

        shifted = hue_shift_red(avg_mat, hue_shift)
        new_avg_r = np.sum(shifted)
        hue_shift += 1
        if hue_shift > MAX_HUE_SHIFT:
            new_avg_r = MIN_AVG_RED

    # Apply hue shift to whole image and replace red channel
    shifted_mat = hue_shift_red(mat, hue_shift)
    new_r_channel = np.sum(shifted_mat, axis=2)
    new_r_channel = np.clip(new_r_channel, 0, 255)
    mat[..., 0] = new_r_channel

    # Get histogram of all channels
    hist_r = hist = cv2.calcHist([mat], [0], None, [256], [0, 256])
    hist_g = hist = cv2.calcHist([mat], [1], None, [256], [0, 256])
    hist_b = hist = cv2.calcHist([mat], [2], None, [256], [0, 256])

    normalize_mat = np.zeros((256, 3))
    threshold_level = (mat.shape[0] * mat.shape[1]) / THRESHOLD_RATIO
    for x in range(256):

        if hist_r[x] < threshold_level:
            normalize_mat[x][0] = x

        if hist_g[x] < threshold_level:
            normalize_mat[x][1] = x

        if hist_b[x] < threshold_level:
            normalize_mat[x][2] = x

    normalize_mat[255][0] = 255
    normalize_mat[255][1] = 255
    normalize_mat[255][2] = 255

    adjust_r_low, adjust_r_high = normalizing_interval(normalize_mat[..., 0])
    adjust_g_low, adjust_g_high = normalizing_interval(normalize_mat[..., 1])
    adjust_b_low, adjust_b_high = normalizing_interval(normalize_mat[..., 2])

    shifted = hue_shift_red(np.array([1, 1, 1]), hue_shift)

    shifted_r, shifted_g, shifted_b = shifted[0][0]

    # print("- shifted_r",shifted_r)
    # print("- shifted_g", shifted_g)
    # print("- shifted_b", shifted_b)

    red_gain = (256 / (adjust_r_high - adjust_r_low))
    green_gain = (256 / (adjust_g_high - adjust_g_low))
    blue_gain = (256 / (adjust_b_high - adjust_b_low))

    # print("*** red_gain", red_gain)
    # print("*** green_gain", green_gain)
    # print("*** blue_gain", blue_gain)

    redOffset = -adjust_r_low / 256 * red_gain
    greenOffset = -adjust_g_low / 256 * green_gain
    blueOffset = -adjust_b_low / 256 * blue_gain

    # print("**** redOffset", redOffset)
    # print("**** greenOffset", greenOffset)
    # print("**** blueOffset", blueOffset)

    adjust_red = shifted_r * red_gain
    #  print("### adjust_red: ",adjust_red)
    adjust_red_green = shifted_g * red_gain
    #  print("### adjust_red_green: ", adjust_red_green)
    adjust_red_blue = (shifted_b * red_gain * BLUE_MAGIC_VALUE)
    #  print("### adjust_red_blue: ",adjust_red_blue)

    return np.array([
        adjust_red, adjust_red_green, adjust_red_blue, 0, redOffset,
        0, green_gain, 0, 0, greenOffset,
        0, 0, blue_gain, 0, blueOffset,
        0, 0, 0, 1, 0,
    ])
###########################################################################
# White balance

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result
###########################################################################

def correct(mat):
    #  print('called correct')
    original_mat = mat.copy()
    filter_matrix = get_filter_matrix(mat)
    corrected_mat = apply_filter(original_mat, filter_matrix)
    corrected_mat = cv2.cvtColor(corrected_mat, cv2.COLOR_RGB2BGR)
    corrected_mat = balance_colors(corrected_mat, 1)
    return corrected_mat


def correct_image(input_path, output_path):
    mat = cv2.imread(input_path)
    rgb_mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)

    corrected_mat = correct(rgb_mat)

    cv2.imwrite(output_path, corrected_mat)

    preview_bfr = mat.copy()
    #width = preview.shape[1] // 2
    #preview[::, width:] = corrected_mat[::, width:]
    preview_ftr= corrected_mat
    preview_bfr = cv2.resize(preview_bfr, (480, 270))
    preview_ftr = cv2.resize(preview_ftr, (480, 270))
    preview_imgs = [cv2.imencode('.png', preview_bfr)[1].tobytes(), cv2.imencode('.png', preview_ftr)[1].tobytes()]
    return preview_imgs


def analyze_video(input_video_path, output_video_path):

    # Initialize new video writer
    cap = cv2.VideoCapture(input_video_path)
    fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
    frame_count = math.ceil(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get filter matrices for every 10th frame
    filter_matrix_indexes = []
    filter_matrices = []
    count = 0

    print("Analyzing...")
    while (cap.isOpened()):

        count += 1
        print(f"{count} frames", end="\r")
        ret, frame = cap.read()
        if not ret:
            # End video read if we have gone beyond reported frame count
            if count >= frame_count:
                break

            # Failsafe to prevent an infinite loop
            if count >= 1e6:
                break

            # Otherwise this is just a faulty frame read, try reading next frame
            continue

        # Pick filter matrix from every N seconds
        if(SAMPLE_SECONDS>0):
            if count % (fps * SAMPLE_SECONDS) == 0:
                mat = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                filter_matrix_indexes.append(count)
                filter_matrices.append(get_filter_matrix(mat))
        else:
            if(SAMPLE_SECONDS==0):
                mat = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                filter_matrix_indexes.append(count)
                filter_matrices.append(get_filter_matrix(mat))
            else:
                if(SAMPLE_SECONDS==-1):
                    if(count==1):
                        mat = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        filter_matrix_indexes.append(count)
                        filter_matrices.append(get_filter_matrix(mat))

        yield count

    cap.release()

    # Build a interpolation function to get filter matrix at any given frame
    filter_matrices = np.array(filter_matrices)

    yield {
        "input_video_path": input_video_path,
        "output_video_path": output_video_path,
        "fps": fps,
        "frame_count": count,
        "filters": filter_matrices,
        "filter_indices": filter_matrix_indexes
    }


def process_video(video_data, yield_preview=False):


    # create colored video path
    video_path_split = video_data["output_video_path"].split("/")
    temp_video_name = "temp_clrd_"+video_path_split[len(video_path_split)-1]
    temp_video_path=temp_dir_path+temp_video_name

    cap = cv2.VideoCapture(video_data["input_video_path"])

    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # video_FourCC = cap.get(cv2.CAP_PROP_FOURCC)
    # video_FourCC = cv2.VideoWriter_fourcc(*'hvc1')
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    new_video = cv2.VideoWriter(temp_video_path, fourcc, video_fps,(int(frame_width), int(frame_height)))

    filter_matrices = video_data["filters"]
    filter_indices = video_data["filter_indices"]

    filter_matrix_size = len(filter_matrices[0])

    def get_interpolated_filter_matrix(frame_number):

        return [np.interp(frame_number, filter_indices, filter_matrices[..., x]) for x in range(filter_matrix_size)]

    print("Processing...")

    frame_count = video_data["frame_count"]

    count = 0
    cap = cv2.VideoCapture(video_data["input_video_path"])
    while (cap.isOpened()):

        count += 1
        percent = 100 * count / frame_count
        print("{:.2f}".format(percent), end=" % \r")
        ret, frame = cap.read()

        if not ret:
            # End video read if we have gone beyond reported frame count
            if count >= frame_count:
                break

            # Failsafe to prevent an infinite loop
            if count >= 1e6:
                break

            # Otherwise this is just a faulty frame read, try reading next
            continue

        # Apply the filter
        rgb_mat = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        interpolated_filter_matrix = get_interpolated_filter_matrix(count)
        corrected_mat = apply_filter(rgb_mat, interpolated_filter_matrix)
        corrected_mat = cv2.cvtColor(corrected_mat, cv2.COLOR_RGB2BGR)
        corrected_mat = balance_colors(corrected_mat, 1)

        new_video.write(corrected_mat)

        if yield_preview:
            preview = frame.copy()
            width = preview.shape[1] // 2
            height = preview.shape[0] // 2
            preview[::, width:] = corrected_mat[::, width:]

            preview = cv2.resize(preview, (480, 270))

            yield percent, cv2.imencode('.png', preview)[1].tobytes()
        else:
            yield None

    cap.release()
    new_video.release()
    call_thread_copy_audio(video_data["input_video_path"], video_data["output_video_path"],temp_video_path)


#####################################################################################
################ color balance correction ###########################################
def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()


def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix


def balance_colors(img, percent=1):
    out_channels = []
    channels = cv2.split(img)
    totalstop = channels[0].shape[0] * channels[0].shape[1] * percent / 200.0
    for channel in channels:
        bc = cv2.calcHist([channel], [0], None, [256], (0, 256), accumulate=False)
        lv = np.searchsorted(np.cumsum(bc), totalstop)
        hv = 255 - np.searchsorted(np.cumsum(bc[::-1]), totalstop)
        lut = np.array([0 if i < lv else (255 if i > hv else round(float(i - lv) / float(hv - lv) * 255)) for i in
                        np.arange(0, 256)], dtype="uint8")
        out_channels.append(cv2.LUT(channel, lut))
    return cv2.merge(out_channels)


################## Adding audio to new video from old video #########################
def copy_audio(inputVideoPath, outputVideoPath,temp_video_path):
    """
    # Create temp dir
    temp_dir='Temp'
    temp_dir_path='./'+ temp_dir+'/'
    """
    #CustomLogger.audio_progress_percentage = 0.0
    #CustomLogger.video_progress_percentage = 0.0

    # Get source video file. Need to extract audio track
    sourceVideo = VideoFileClip(inputVideoPath)
    #print("inputVideoPath: "+inputVideoPath)

    # Set colored videofile. Need to merge with extracted audio track
    coloredVideo = VideoFileClip(temp_video_path)
    #print("outputVideoPath: " + outputVideoPath)

    # set temp sounded video path
    #get name of video file
    splitted_path_array = inputVideoPath.split("/")
    in_filename = splitted_path_array[len(splitted_path_array)-1]
    #print("in_filename: " + in_filename)
    soundedVideoPath = temp_dir_path+"temp_"+in_filename
    #print("soundedVideoPath: " + soundedVideoPath)

    # set temp audio file name and path
    audio = 'temp_'+in_filename.replace("mp4", "mp3").replace("MP4","mp3")
    audio_path = temp_dir_path+audio
    #print("audio_path: " + audio_path)

    #define source audiofile
    source_audiofile=sourceVideo.audio

    # init params before calling audio processing
    #CustomLogger.audio_progress_percentage=0.0
    #CustomLogger.video_progress_percentage=0.0


    # write audio to temp dir
    source_audiofile.write_audiofile(audio_path,logger=audio_proc_logger)   # loading recorded audio file
    audioclip = AudioFileClip(audio_path)   # Set audio from source video to final video
    final_clip = coloredVideo.set_audio(audioclip)   # Write final video file
    final_clip.write_videofile(soundedVideoPath, sourceVideo.fps, logger=video_proc_logger)

    try:
        # remove audio file from temp dir
        if(os.path.isfile(audio_path)):
            os.remove(audio_path)
    except:
        print("Error with access to file. Can't delete file: " + audio_path)

    try:
        # remove init video file from temp dir
        if (os.path.isfile(temp_video_path)):
            os.remove(temp_video_path)

    except:
        print("Error with access to file. Can't delete file: " + temp_video_path)

    try:
        # replace final video from temp
        if (os.path.isfile(soundedVideoPath)):
            os.replace(soundedVideoPath, outputVideoPath)

    except:
        print("Error with access to file. Can't move file " + soundedVideoPath + " to " + outputVideoPath)



    CustomLogger.audio_progress_percentage = 0.0
    CustomLogger.video_progress_percentage = 0.0

def call_thread_copy_audio(inputVideoPath, outputVideoPath,temp_video_path):
    """
    thread_copy_audio = threading.Thread(target=copy_audio)
    thread_copy_audio.daemon = True
    thread_copy_audio.args = (inputVideoPath, outputVideoPath)
    thread_copy_audio.start()
    """
    thread_01 = threading.Thread(target=copy_audio, args=(inputVideoPath,outputVideoPath,temp_video_path,))
    thread_01.start()
#####################################################################################
# if __name__ == "__main__":
# T###################Test for photo##################################
# inputImage = "./data/f0664064.jpg"
# outputImage = "./data/out_f0664064.jpg"
# mat = cv2.imread(inputImage)
# mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
# corrected_mat = correct(mat)
################ Contrast and Brightness setting ###################
# alpha = 1.1 # Contrast control (1.0-3.0)
# beta = -1 # Brightness control (0-100)
# corrected_mat = cv2.convertScaleAbs(corrected_mat, alpha=alpha, beta=beta)
####################################################################
# cv2.imwrite(outputImage, corrected_mat)

################ Test for video ##############################
""""    
inputVideo = "D:/Color correction test/YDXJ0084.mp4"
outputVideo = "D:/Color correction test/YDXJ0084_corrected.mp4"

for item in analyze_video(inputVideo, outputVideo):
    if type(item) == dict:
        video_data = item
[x for x in process_video(video_data, yield_preview=False)]

"""
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage")
        print("-" * 20)
        print("For image:")
        print("$python correct.py image <source_image_path> <output_image_path>\n")
        print("-" * 20)
        print("For video:")
        print("$python correct.py video <source_video_path> <output_video_path>\n")
        exit(0)

    if (sys.argv[1]) == "image":
        mat = cv2.imread(sys.argv[2])
        mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)

        corrected_mat = correct(mat)

        cv2.imwrite(sys.argv[3], corrected_mat)

    else:

        for item in analyze_video(sys.argv[2], sys.argv[3]):

            if type(item) == dict:
                video_data = item

        [x for x in process_video(video_data, yield_preview=False)]
