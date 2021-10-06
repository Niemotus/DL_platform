from Queue import Queue
from threading import Thread
import random as rnd
import nibabel as nib
import numpy as np
import Common


class BatchGenerator:
    """
        This class implements an interface for batch generators. 

        In one epoch we process several subepochs, every one formed by multiple batches
        (every batch is processed independently by the CNN). In every subepoch, new volumes are read from disk
        and used to sample segments for several batches.

        The method generateBatches( ) runs a thread that will call the method generateBatchesForOneEpoch( ) as many
        times as confTrain['numEpochs'] indicates. The method generateBatchesForOneEpoch( ) runs several subepochs,
        reading new volumes in every subepoch. In every subepoch, this method generates several batches made of
        different random segments extracted from the loaded volumes.
        The batches will be all queued in self.queue using self.queue.put(batch).
    """

    def __init__(self, confTrain, maxQueueSize = 20):
        """
            Creates a batch generator.

            :param confTrain: a configuration dictionary containing the necessary training parameters
            :param maxQueueSize: maximum number of batches that will be inserted in the queue at the same time.
                           If this number is achieved, the batch generator will wait until one batch is
                           consumed to generate a new one.

                           The number of elements in the queue can be monitored using getNumBatchesInQueue. The queue
                           should never be empty so that the GPU is never idle. Note that the bigger maxQueueSize,
                           the more RAM will the program consume to store the batches in memory. You should find
                           a good balance between RAM consumption and keeping the GPU processing batches all the time.


            :return: self.queue.empty()
        """
        self.confTrain = confTrain
        self.queue = Queue(maxsize=maxQueueSize)

    def emptyQueue(self):
        """
            Checks if the batch queue is empty or not.

            :return: self.queue.empty()
        """

        return self.queue.empty()

    def _generateBatches(self):
        """
            Private function that generates as many batches as epochs were specified
        """
        for e in range(0, self.confTrain['numEpochs']):
            self.generateBatchesForOneEpoch()

    def generateBatches(self):
        """
            This public interface lunches a thread that will start generating batches for the epochs/subepochs specified
            in the configuration file, and storing them in the self.queue.
            To extract these batches, use self.getBatch()
        """
        worker = Thread(target=self._generateBatches, args=())
        worker.setDaemon(True)
        worker.start()

    def getBatch(self):
        """
            It returns a batch and removes it from the front of the queue

            :return: a batch from the queue
        """
        batch = self.queue.get()
        self.queue.task_done()
        return batch

    def getNumBatchesInQueue(self):
        """
            It returns the number of batches currently in the queue, which are ready to be processed

            :return: number of batches in the queue
        """
        return self.queue.qsize()

    def generateBatchesForOneEpoch(self):
        """
            This abstract function must be implemented. It must generate all the batches corresponding to one epoch
            (one epoch is divided in subepochs where different data samples are read from disk, and every subepoch is
            composed by several batches, where every batch includes many segments.)

            Every batch must be queued using self.queue.put(batch) and encoded using lasagne-compatible format,
            i.e: a 5D tensor with size (batch_size, num_input_channels, input_depth, input_rows, input_columns)

        """
        raise NotImplementedError('users must define "generateBatches" to use this base class')



def checkFlip(array, flip):
    if flip:
        return np.flipud(array)
    else:
        return array



class Simple2DBatchGenerator(BatchGenerator):
    """
        Simple batch generator that takes multi-channel 2D images with labels and masks,
        and samples random patches.
    """

    def __init__(self, confTrain):
        """
            Initialize a 2D batch generator with a confTrain object. The batch generator
            will load the training files (channelsTraining, gtLabelsTraining)
            from the confTrain object.
        """

        BatchGenerator.__init__(self, confTrain)

        # List of currently loaded channel images
        self.currentChannelImages = []
        # List of currently loaded GT images
        self.currentGt = []
        # List of list of labels available in currently loaded GT images
        self.availableLabels = []


        # ===================================== ABOUT THE IMAGE CHANNELS LIST INDEXING     ================================

        # IMPORTANT: The indices in the lists of channel filenames are inverted with respect to the currently loaded images
        # self.currentChannelImages is [img][channel] while the filenames list self.allChannelsFilenames is [channel][img]

        # =================================================================================================================

        # List of lists of filenames (size: numChannels x numOfFilenames)
        self.allChannelsFilenames = []
        # List of GT filenames (size:numOfFilenames)
        self.gtFilenames = []

        self.allowedIndicesPerVolumePerLabel = {}

        self.numClasses =  len(self.confTrain['labels'])

        # Load the filenames from the configuration files and the parameters
        self.numChannels = len(self.confTrain['channelsTraining'])
        self.numOfCasesLoadedPerSubepoch = self.confTrain['numOfCasesLoadedPerSubepochTraining']
        self.loadFilenames(self.confTrain['channelsTraining'], self.confTrain['gtLabelsTraining'])

        # Check if, given the number of cases loaded per subepoch and the total number of samples, we
        # will need to load new data in every subepoch or not.
        self.loadNewFilesEverySubepoc = len(self.allChannelsFilenames[0]) > self.numOfCasesLoadedPerSubepoch


        self.tileSize = self.confTrain['patchSizeTrain']
        self.gtSize = self.confTrain['patchSizeGt']


    def loadFilenames(self, channels, gtLabels):
        """
            Load the filenames that will be used to generate the batches.

            :param channels: list containing the path to the text files containing the path to the channels (this list
                            contains one file per channel).
            :param gtLabels: path of the text file containing the paths to gt label files

        """
        self.allChannelsFilenames, self.gtFilenames = Common.loadFilenames(channels, gtLabels)


    def unloadFiles(self):
        for image in self.currentChannelImages:
            for channelImage in image:
                del channelImage

        for image in self.currentGt:
            del image

        del self.currentChannelImages
        del self.currentGt
        del self.availableLabels

        for k in self.allowedIndicesPerVolumePerLabel.keys():
            del self.allowedIndicesPerVolumePerLabel[k]

        del self.allowedIndicesPerVolumePerLabel

        self.currentChannelImages = []
        self.currentGt = []
        self.availableLabels = []
        self.allowedIndicesPerVolumePerLabel = {}


    def generateRandomSegmentCenteredAtLabel(self, volumeToSample, labelToSample):
        """
            Generates a 2D segment fom 3D volume and the corresponding Ground truth
            :param volumeToSample: indicates the volume number (from the current ones) that will be sampled
            :param labelToSample: indicates in which label must the segment be centered

            :return: returns a 2D segment and the corresponding ground truth
        """
        labelCandidates = self.allowedIndicesPerVolumePerLabel[volumeToSample][labelToSample]
        voxelIndex = labelCandidates[rnd.randint(0, labelCandidates.shape[0]-1),:]

        tileOffset = [x // 2 for x in self.tileSize]#self.tileSize // 2
        gtOffset = [x // 2 for x in self.gtSize]#self.gtSize // 2

        data = np.ndarray(shape=(1, self.numChannels, self.tileSize[0], self.tileSize[1]), dtype=np.float32)
        gt =   np.ndarray(shape=(1, self.numClasses,  self.gtSize[0],   self.gtSize[1]),   dtype=np.float32)



        # Check if we have to flip the segment
        flip = np.random.binomial(1, self.confTrain["randomlyFlipSegmentsProbability"], 1)[0] == 1


        indx = [[voxelIndex[0],voxelIndex[0]+1] if tileOffset[0] == 0 else [voxelIndex[0]-tileOffset[0], voxelIndex[0]+tileOffset[0]]][0]
        indy = [[voxelIndex[1],voxelIndex[1]+1] if tileOffset[1] == 0 else [voxelIndex[1]-tileOffset[1], voxelIndex[1]+tileOffset[1]]][0]
        indz = voxelIndex[2]
        for channel in range(self.numChannels):
            data[0, channel, :, :] = checkFlip(self.currentChannelImages[volumeToSample][channel].get_data()[indx[0]:indx[1], indy[0]:indy[1], indz], flip)

        # Crop the area corresponding to the GT
        indx = [[voxelIndex[0], voxelIndex[0] + 1] if gtOffset[0] == 0 else [voxelIndex[0] - gtOffset[0], voxelIndex[0] + gtOffset[0]]][0]
        indy = [[voxelIndex[1], voxelIndex[1] + 1] if gtOffset[1] == 0 else [voxelIndex[1] - gtOffset[1], voxelIndex[1] + gtOffset[1]]][0]
        indz = voxelIndex[2]
        auxGt = checkFlip(self.currentGt[volumeToSample].get_data()[indx[0]:indx[1], indy[0]:indy[1], indz], flip)

        # Transform the GT in a format that can be read
        for i in range(self.numClasses):
            gt[0, i, :, :] = (auxGt==self.confTrain['labels'][i]).astype(np.int16)

        return data, gt


    def generateSingleBatch(self):
        """
            Creates a batch of segments according to the conf file. It supposes that the images are already
            loaded in self.currentVolumes.

            :return: It returns the data and ground truth of a complete batch as data, gt. These structures are theano-compatible with shape:
                        np.ndarray(shape=(self.confTrain['batchSizeTraining'], self.numChannels, tileSize, tileSize, tileSize), dtype=np.float32)
        """
        batchSize = self.confTrain['batchSizeTraining']

        batch = np.ndarray(shape=(batchSize, self.numChannels, self.tileSize[0], self.tileSize[1]), dtype=np.float32)
        gt =    np.ndarray(shape=(batchSize, self.numClasses,  self.gtSize[0],   self.gtSize[1]),   dtype=np.float32)


        # Number of segments that will be centered at a positive sample
        numCasesPerLabel = (batchSize * self.percentOfSamplesPerClass).astype('int')
        while not np.sum(numCasesPerLabel)==self.confTrain['batchSizeTraining']:
            numCasesPerLabel[rnd.randint(1, np.max(self.labels))] += 1

        if not(self.currentChannelImages == []):
            # Extract samples (patches) according to numCasesPerLabel
            cont = 0
            for numCases, lab in zip(numCasesPerLabel, self.labels):
                for i in range(0, numCases):
                    # Choose a random volume to sample from that contains the rigth label
                    volumeToSampleFrom = rnd.randint(0, self.numOfCasesLoadedPerSubepoch - 1)
                    while not lab in self.availableLabels[volumeToSampleFrom]:
                        volumeToSampleFrom = rnd.randint(0, self.numOfCasesLoadedPerSubepoch - 1)

                    # Generate a new  2D segment (data and ground truth) and add it to the batch
                    batch[cont, :, :, :], gt[cont, :, :, :] = self.generateRandomSegmentCenteredAtLabel(volumeToSampleFrom, lab)
                    cont += 1

            # Return the batch data with its corresponding ground truth
            return batch, gt
        else:
            raise Exception(self.id + " No images loaded in self.currentVolumes." )


    def generateBatchesForOneEpoch(self):
        for se in range(self.confTrain['numSubepochs']):
            if (se == 0) or self.loadNewFilesEverySubepoc:
                # Choose the random images that will be sampled in this epoch
                indexCurrentImages = rnd.sample(xrange(0, len(self.allChannelsFilenames[0])), self.numOfCasesLoadedPerSubepoch)

                # Load the images for the epoch
                i = 0
                for realImageIndex in indexCurrentImages:
                    # print(i)
                    # Load GT for the current image
                    gt = nib.load(self.gtFilenames[realImageIndex])

                    # For every channel

                    loadedImageChannels = []
                    for channel in range(0, len(self.allChannelsFilenames)):
                        # Load the corresponding image for the corresponding channel and append it to the list of channels
                        # for the current imageIndex
                        image = nib.load(self.allChannelsFilenames[channel][realImageIndex])
                        loadedImageChannels.append(image)

                    # Append all the channels of the image to the list
                    self.currentChannelImages.append(loadedImageChannels)

                    # Append GT to the list
                    self.currentGt.append(gt)
                    self.availableLabels.append(list(np.unique(gt.get_data()).astype('int')))


                    # Filter pixel positions by label.
                    self.labels = self.confTrain['labels'] #np.unique(gt.get_data())

                    self.allowedIndicesPerVolumePerLabel[i] = {}

                    # Create a mask to restrict the indices to valid limits
                    validArea = np.zeros(shape=gt.get_data().shape)
                    tileOffset = [x // 2 for x in self.tileSize] #self.tileSize // 2
                    # tileOffset = [None if x is 0 else x for x in tileOffset]
                    validArea[tileOffset[0]:(gt.get_data().shape[0]-tileOffset[0]),\
                              tileOffset[1]:(gt.get_data().shape[1]-tileOffset[1]),\
                              tileOffset[2]:(gt.get_data().shape[2]-tileOffset[2])] = 1

                    for indL in range(len(self.labels)):
                        self.allowedIndicesPerVolumePerLabel[i][indL] = np.argwhere((gt.get_data() == self.labels[1]) & (validArea == 1))

                    del validArea
                    i += 1

            # Labels available in subset of images loaded in current subepoch. If a label is unavailable it will be ignored
            self.subEpochLabels = [L in [y for x in self.availableLabels for y in x] for L in self.confTrain['labels']]
            self.labels = [i for indx,i in enumerate(self.labels) if self.subEpochLabels[indx] == True]

            # redistribute percentages according to available labels in subepoch
            self.percentOfSamplesPerClass = self.confTrain['percentOfSamplesPerClass']
            self.percentOfSamplesPerClass = [i for indx,i in enumerate(self.percentOfSamplesPerClass) if self.subEpochLabels[indx] == True]
            self.percentOfSamplesPerClass = np.array(self.percentOfSamplesPerClass)/np.sum(np.array(self.percentOfSamplesPerClass)).astype('float')

            # Generate batches and put in queue
            for batch in range(0, self.confTrain['numBatchesPerSubepoch']):
                data, gt = self.generateSingleBatch()
                self.queue.put((data,gt))

            # Unload the files if we are in the last subepoch or if we are loading new files every subpeoc
            if self.loadNewFilesEverySubepoc or (se == self.confTrain['numSubepochs'] - 1):
                self.unloadFiles()

