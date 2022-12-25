from proglog import ProgressBarLogger
video_progress_percentage=0.0
audio_progress_percentage=0.0

class VideoLogger(ProgressBarLogger):
    def bars_callback(self, bar, attr, value, old_value=None, **changes):
        global video_progress_percentage
        # Every time the logger progress is updated, this function is called
        if len(self.bars):

            if(next(reversed(self.bars.items()))[1]['indent']==2):
                video_progress_percentage=next(reversed(self.bars.items()))[1]['index']/next(reversed(self.bars.items()))[1]['total']
                if(video_progress_percentage>=0):
                    video_progress_percentage=video_progress_percentage*100


video_proc_logger = VideoLogger()
class AudioLogger(ProgressBarLogger):
    def bars_callback(self, bar, attr, value, old_value=None):
        global audio_progress_percentage
        # Every time the logger progress is updated, this function is called
        percentage = (value / self.bars[bar]['total']) * 100
        audio_progress_percentage=round(percentage,2)

audio_proc_logger = AudioLogger()



