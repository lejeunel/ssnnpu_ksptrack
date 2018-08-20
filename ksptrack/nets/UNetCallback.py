from keras.callbacks import Callback
import csv
import os

class UNetCallback(Callback):
    def __init__(self, log_file_path, weight_file_path):
        """
        Initialize model callback class

        :param log_file_path:    Full path to the log file
        :param weight_file_path: Full path to the weight file
        :return:                 None
        """
        super(UNetCallback, self).__init__()

        self.logFilePath = log_file_path
        self.weightFilePath = weight_file_path
        self.isFirstRun = True


    def on_epoch_end(self, epoch, logs=None):
        """
        This method is called at every epoch end and logs the metrics into a csv

        :param epoch: Current number of epochs
        :param logs:  The logs as a dictionary
        :return:      None
        """
        isWriteHeader = False

        if self.isFirstRun:
            self.isFirstRun = False

            # Change path if already exist
            if os.path.isfile(self.logFilePath):
                self.logFilePath = get_nonexistant_path(self.logFilePath)
                #os.remove(self.logFilePath)
            isWriteHeader = True

        with open(self.logFilePath, mode='a') as csvfile:
            csvWriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            if isWriteHeader:
                headerList = list()
                headerList.append('epoch')
                rowList = list()
                rowList.append(epoch)
                for key, val in sorted(logs.items()):
                    headerList.append(key)
                    rowList.append(val)
                csvWriter.writerow(headerList)
                csvWriter.writerow(rowList)
            else:
                rowList = list()
                rowList.append(epoch)
                for _, val in sorted(logs.items()):
                    rowList.append(val)
                csvWriter.writerow(rowList)


def get_nonexistant_path(fname_path):
    """
    Get the path to a filename which does not exist by incrementing path.

    Examples
    --------
    >>> get_nonexistant_path('/etc/issue')
    '/etc/issue-1'
    >>> get_nonexistant_path('whatever/1337bla.py')
    'whatever/1337bla.py'
    """
    if not os.path.exists(fname_path):
        return fname_path
    filename, file_extension = os.path.splitext(fname_path)
    i = 1
    new_fname = "{}-{}{}".format(filename, i, file_extension)
    while os.path.exists(new_fname):
        i += 1
        new_fname = "{}-{}{}".format(filename, i, file_extension)
    return new_fname
