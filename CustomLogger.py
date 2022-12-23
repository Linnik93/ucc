from proglog import ProgressBarLogger
progress_percentage=0.0
audio_progress_percentage=0.0

class VideoLogger(ProgressBarLogger):
    def bars_callback(self, bar, attr, value, old_value=None):
        global progress_percentage
        # Every time the logger progress is updated, this function is called
        percentage = (value / self.bars[bar]['total']) * 100
        progress_percentage=round(percentage,2)

video_proc_logger = VideoLogger()
class AudioLogger(ProgressBarLogger):
    def bars_callback(self, bar, attr, value, old_value=None):
        global audio_progress_percentage
        # Every time the logger progress is updated, this function is called
        percentage = (value / self.bars[bar]['total']) * 100
        audio_progress_percentage=round(percentage,2)

audio_proc_logger = AudioLogger()



