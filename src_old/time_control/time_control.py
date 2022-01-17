class TimeControl:
    def __init__(self, time_control_block):
        self.time_control_block = time_control_block

        self.time_start = self.time_control_block['time_start']
        self.time_end = self.time_control_block['time_end']
        self.time_increment = self.time_control_block['time_increment']

        self.t = self.time_start
        self.time_step_number = 1  # start from one since exodus is 1 based

    def __str__(self):
        string = '----- Time Control -----\n'
        string = string + \
                 'Time start     = %s\n' % self.time_start + \
                 'Time end       = %s\n' % self.time_end + \
                 'Time increment = %s\n' % self.time_increment + \
                 'Time current   = %s\n' % self.t
        return string

    def increment_time(self):
        self.t = self.t + self.time_increment
        self.time_step_number = self.time_step_number + 1
