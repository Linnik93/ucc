from proglog import ProgressBarLogger
video_progress_percentage=0.0
audio_progress_percentage=0.0
proc_stage=0
trig=0


class VideoLogger(ProgressBarLogger):
    def bars_callback(self, bar, attr, value, old_value=None, **changes):
        global video_progress_percentage
        global proc_stage
        global trig
        # Every time the logger progress is updated, this function is called
        if len(self.bars):
            if((next(reversed(self.bars.items()))[1]['index']/next(reversed(self.bars.items()))[1]['total']) * 100>=100):
                proc_stage=proc_stage+1
                print("Proc stage: ",proc_stage)
            if(proc_stage>=1 and (next(reversed(self.bars.items()))[1]['index']/next(reversed(self.bars.items()))[1]['total']) * 100<0):
                trig=1
            if(trig==1):
                video_progress_percentage = (next(reversed(self.bars.items()))[1]['index']/next(reversed(self.bars.items()))[1]['total']) * 100
                video_progress_percentage = round(video_progress_percentage, 2)
                print("video_progress_percentage: ",video_progress_percentage);


        """
        if((value / self.bars[bar]['total']) * 100>=100.00):
            cache_percentage = (value / self.bars[bar]['total']) * 100
        if(cache_percentage>=100):
            cache_percentage2 = 100.00

        #video_progress_percentage=round(percentage,2)
        if len(self.bars):
            if(cache_percentage2>=100.00):
                video_progress_percentage = (next(reversed(self.bars.items()))[1]['index'] / next(reversed(self.bars.items()))[1]['total'])*100
                video_progress_percentage = round(video_progress_percentage, 2)
                print("############## TEST ",round(video_progress_percentage, 2))
        """


video_proc_logger = VideoLogger()
class AudioLogger(ProgressBarLogger):
    def bars_callback(self, bar, attr, value, old_value=None):
        global audio_progress_percentage
        # Every time the logger progress is updated, this function is called
        percentage = (value / self.bars[bar]['total']) * 100
        audio_progress_percentage=round(percentage,2)

audio_proc_logger = AudioLogger()



