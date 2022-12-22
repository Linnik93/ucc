from proglog import ProgressBarLogger
class MyBarLogger(ProgressBarLogger):
    def bars_callback(self, bar, attr, value, old_value=None):
        # Every time the logger progress is updated, this function is called
        percentage = (value / self.bars[bar]['total']) * 100
        print(percentage)

logger = MyBarLogger()