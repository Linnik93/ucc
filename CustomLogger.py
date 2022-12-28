from proglog import ProgressBarLogger
video_progress_percentage=0.0
audio_progress_percentage=0.0

prev_percentage = 0
preproc_index=0

class VideoLogger(ProgressBarLogger):
    def bars_callback(self, bar, attr, value, old_value=None, **changes):
        global video_progress_percentage,preproc_index

        # Every time the logger progress is updated, this function is called

        if (next(reversed(self.bars.items()))[1]['index']/next(reversed(self.bars.items()))[1]['total']*100>=100 ):
            preproc_index = next(reversed(self.bars.items()))[1]['indent']

        if (next(reversed(self.bars.items()))[1]['indent']>=preproc_index):
            if(next(reversed(self.bars.items()))[1]['index']/next(reversed(self.bars.items()))[1]['total']*100<100 and next(reversed(self.bars.items()))[1]['index']/next(reversed(self.bars.items()))[1]['total']*100>0 and next(reversed(self.bars.items()))[0]!='chunk'):
                video_progress_percentage=(next(reversed(self.bars.items()))[1]['index']/next(reversed(self.bars.items()))[1]['total']*100)


video_proc_logger = VideoLogger()
class AudioLogger(ProgressBarLogger):
    def bars_callback(self, bar, attr, value, old_value=None):
        global audio_progress_percentage
        # Every time the logger progress is updated, this function is called
        percentage = (value / self.bars[bar]['total']) * 100
        audio_progress_percentage=round(percentage,2)

audio_proc_logger = AudioLogger()



