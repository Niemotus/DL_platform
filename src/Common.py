import os

def getImageBasename(filename):
    if filename.endswith(".nii.gz"):
        return os.path.splitext(os.path.splitext(os.path.split(filename)[1])[0])[0]
    else:
        return os.path.splitext(os.path.split(filename)[1])[0]

def loadFilenames(channels, gtLabels):
    """
        Load the filenames files (for training, validation or testing)

        :param channels: list containing the path to the text files containing the path to the channels (this list
                        contains one file per channel).
        :param gtLabels: path of the text file containing the paths to gt label files

        :return allChannelsFilenames, gtFilenames
                allChannelsFilenames is a list of lists where allChannelsFilenames[0] contains a list of filenames for channel 0 (size: numChannels x numFiles)
                gtFilenames is a list of GT filenames (size: numFiles)
               

    """
    allChannelsFilenames = []
    gtFilenames = None

    # Load training filenames
    if len(channels) > 0:
        # For every channel
        for filenames in channels:
            # Load the filenames for the given channel
            with open(filenames) as f:
                allChannelsFilenames.append([line.rstrip('\n') for line in f if line.rstrip('\n') != ""])

        for i in range(1, len(channels)):
            assert len(allChannelsFilenames[i]) == len(allChannelsFilenames[0]), "[ERROR] All the channels must contain same number of filenames"

        # Load the GT filenames [Required]
        with open(gtLabels) as f:
            gtFilenames = [line.rstrip('\n') for line in f if line.rstrip('\n') != ""]

        assert len(gtFilenames) == len(allChannelsFilenames[0]), "[ERROR] Number of GT filenames must be the same than image (channels) filenames"
    else:
        raise Exception("channel files are missing")

    return allChannelsFilenames, gtFilenames