import sys, os, time
import argparse
import nibabel as nib
import numpy as np
import theano
import theano.tensor as T
import lasagne
from network import build_uResNet
from BatchGenerator import Simple2DBatchGenerator
from Common import loadFilenames
from helper_functions import categorical_crossentropy, calculate_dice, sorenson_dice

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--Train", type=int, help="Training dataset to use")
    parser.add_argument('--init', metavar='file', nargs=1, help='initial weights for the model')
    parser.add_argument('--init_n', metavar='int', nargs=1, help='initial epochs')
    parser.add_argument('--input_chan', type=str, nargs=1, help='input channels to use')
    parser.add_argument("--loss_name",  type=str, help="Loss name")
    parser.add_argument("--seed", type=int, help="random seed value: default is 10")
    args = parser.parse_args()

    # PARAMETERS
    loss_name     = ''
    lr_reduce     = False
    debug         = False
    regularize    = False
    seed          = 10
    weight_decay = 1e-4

    if args.seed:
        seed = args.seed
    np.random.seed(seed)

    if args.Train:
        train=args.Train
    else:
        train=1
    if args.loss_name:
        loss_name=args.loss_name
    print("Training dataset: \t{:d}".format(train))

    base_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))

    # Load the traininf conf file (in this file the test set should be defined)
    confTrain = {}
    execfile(base_path + "/conf/training_dataset"+str(train)+".conf", confTrain)

    # Create and start the batch generator (this creates a CPU process that generates batches in the background)
    batchGen = Simple2DBatchGenerator(confTrain)
    batchGen.generateBatches()


    # Input channels
    print('Input channel lists:')
    for i in confTrain['channelsTraining']:
        print('\t'+i)
    learning_rate = theano.shared(np.array(confTrain['learningRate'], dtype='float32'))
    num_epochs    = confTrain["numEpochs"]
    batch_size    = confTrain["batchSizeTraining"]
    label_num     = confTrain["labels"]

    # Load the training and testing image file names
    imageFilesTrain, labelFilesTrain = loadFilenames(
        confTrain['channelsTraining'], confTrain['gtLabelsTraining'])
    imageFilesTest, labelFilesTest = loadFilenames(
        confTrain['channelsValidation'], confTrain['gtLabelsValidation'])

    # Get input channels from file name
    input_chan = [imageFilesTrain[i][0].split('/')[-1].split('.')[-3] for i in range(len(imageFilesTrain))]

    # Build the network
    net_name = 'uResNet3D'
    # Prepare theano variables
    image_var = T.tensor4('image')
    label_var = T.itensor4('label')

    net = build_uResNet(image_var, shape=(None, len(confTrain['channelsTraining']), None, None), n_class=len(confTrain['labels']))
    print("Build network:\t\t{:s}".format(net_name))
    print("L2 regularization: \t"+str(regularize))

    # Load the dataset
    print("Loading data...")

    # For debug
    if debug:
        print('Running in DEBUG...')
        net_name = 'DEBUG_'+net_name
        mode = theano.Mode(optimizer = 'fast_compile')
    else:
        mode = theano.Mode(optimizer = 'fast_run')

    # If initial model provided, load weights
    if args.init:
        model_file = args.init[0]
        print('Initialise the convolutional layers using weights from {0} ...'.format(model_file))
        with np.load(model_file) as f:
            param_values = [f['arr_{0}'.format(i)] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(net['output'], param_values)

    # Loss function
    print("Loss function...")
    prediction = lasagne.layers.get_output(net['output'])
    if   loss_name=='':
        loss = categorical_crossentropy(prediction, label_var)
        print("\t\tCross entrophy...")
    elif loss_name=='_Dice':
        loss = sorenson_dice(prediction, label_var, 1) + sorenson_dice(prediction, label_var, 2) + sorenson_dice(prediction, label_var, 0)
        print("\t\tDice score...")
    else:
        sys.exit('UNDEFINED LOSS!!!')

    # Update expression
    print("Create update expression for training...")

    if regularize:
        weightsl2 = lasagne.regularization.regularize_network_params(net['output'], lasagne.regularization.l2)
        loss += weight_decay * weightsl2
        #loss = loss + lasagne.regularization.regularize_layer_params_weighted(net['output'], lasagne.regularization.l2)

    params = lasagne.layers.get_all_params(net['output'], trainable=True)
    updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)

    # Create an expression for testing. The crucial difference here is that we do a deterministic
    # forward pass through the network, disabling dropout layers.
    print("Create an expression for testing...")
    test_prediction = lasagne.layers.get_output(net['output'], deterministic=True)

    # Compile a function performing a training step on a mini-batch and returning the corresponding training loss
    print("Compile a training function...")
    train_fn = theano.function([image_var, label_var], loss, updates=updates, mode=mode)

    # Compile a second function that segments images:
    print("Compile a segmentation function...")
    segment_fn = theano.function([image_var], test_prediction, mode=mode)

    # Prepare the logger
    print("Prepare the logger...")
    if not args.init:
        csv_name = os.path.join(base_path, 'logs/{0}_log_dataset{1}_ch-{2}{3}_seed{4}_L2-{5}.csv'.format(net_name,train, '-'.join(str(s) for s in input_chan),loss_name, seed, str(regularize)))
        f_log = open(csv_name, 'w')
        f_log.write('epoch,lr_rate,time,train_loss,test_dice,train_dice\n')
    else:
        csv_name = os.path.join(base_path, 'logs/{0}_log_dataset{1}_ch-{2}{3}_seed{4}_L2-{5}.csv'.format(net_name,train, '-'.join(str(s) for s in input_chan),loss_name, seed, str(regularize)))
        f_log = open(csv_name, 'a')


    # Launch the training loop
    print("Starting training...")
    start_time = time.time()
    table = []

    if args.init_n:
        start = 1 + int(args.init_n[0])
    else:
        start = 1
    for epoch in range(start, start + num_epochs):
        # In each epoch, we do a full pass over the training data:
        start_time_epoch = time.time()
        train_loss = 0
        train_batches = 0
        for i in range(0, confTrain['numBatchesPerSubepoch']*confTrain['numSubepochs']):
            image, label = batchGen.getBatch()
            # print(batchGen.getNumBatchesInQueue())
            train_loss += train_fn(image, label.astype('int16'))
            train_batches += 1
        train_loss /= train_batches

        test_dice1 = 0
        test_images1 = 0
        train_dice1 = 0
        train_images1 = 0

        # Save model, and full pass over the testing and testing data every "epochsSaveTestEvery" defined in conf file
        if ((epoch % confTrain["epochsSaveTestEvery"]) == 0):

            # Store model:
            model_path = os.path.join(base_path, 'model')
            np.savez(os.path.join(model_path, '{0}_DATASET{1}_ch-{2}_epoch{3:03d}{4}_seed{5}_L2-{6}.npz'.format(
                net_name, train, '-'.join(str(s) for s in input_chan), epoch, loss_name, seed, str(regularize))), *lasagne.layers.get_all_param_values(net['output']))



            # Check the dice on whole images (train and test)
            print("Testing epoch started...")
            for data in range(len(imageFilesTest[0])):

                # Load input channels
                for num_chan in range(len(imageFilesTest)):
                    image_name = imageFilesTest[num_chan][data]
                    image_data = nib.load(image_name)

                    # Add additional dimension corresponging to the number of channels
                    image = np.expand_dims(image_data.get_data().astype('float32'), axis=0)
                    if num_chan == 0:
                        images = image
                    else:
                        images = np.concatenate((images, image), axis=0)
                # Add additional dimension corresponging to the number of batches (one in this case)
                images = np.expand_dims(images, axis=0)

                # Crop image to minimum box of non-background to save memory and avoid computations over the background
                inds_max = np.max(np.where(images[0, 0, :, :, :] > -3), axis=1)
                inds_min = np.min(np.where(images[0, 0, :, :, :] > -3), axis=1)
                images = images[:, :, inds_min[0]:inds_max[0] + 1, inds_min[1]:inds_max[1] + 1, inds_min[2]:inds_max[2] + 1]

                # Pad image with zeros so that dimensions are multiples of 8. This is necesary due to the architecture
                pad = [(0, int(np.ceil(np.array(images.shape[i]) / 8.0) * 8 - np.array(images.shape[i])))
                       if i > 1 else (0, 0) for i in range(0, 5)]
                images = np.pad(images, pad, mode='constant', constant_values=-3)

                # Load labels and cut to min
                label_name = labelFilesTest[data]

                label_data = nib.load(label_name)
                gt_label = label_data.get_data().astype('int')
                gt_label = gt_label[inds_min[0]:inds_max[0] + 1, inds_min[1]:inds_max[1] + 1, inds_min[2]:inds_max[2] + 1]

                # Segment image


                pred = np.zeros((1,len(label_num),len(images[0,0]),len(images[0,0,0]), len(images[0,0,0,0])))
                for slice in range(len(images[0, 0, 0, 0])):
                    pred[:, :, :, :, slice] = segment_fn(images[:, :, :, :, slice])


                # Remove padding from prediction
                pred = np.squeeze(pred[:pred.shape[0] - pad[0][1], :pred.shape[1] - pad[1][1], :pred.shape[2] - pad[2][1], :pred.shape[3] - pad[3][1], :pred.shape[4] - pad[4][1]], axis=0)


                # Claculate Dice scores
                dice1 = calculate_dice(pred, gt_label, 1)
                test_dice1 += dice1
                test_images1 += 1
            test_dice1 /= test_images1


            for data in range(len(imageFilesTrain[0])):

                # Load input channels
                for num_chan in range(len(imageFilesTrain)):
                    image_name = imageFilesTrain[num_chan][data]
                    image_data = nib.load(image_name)
                    # Add additional dimension corresponging to the number of channels
                    image = np.expand_dims(image_data.get_data().astype('float32'), axis=0)
                    if num_chan == 0:
                        images = image
                    else:
                        images = np.concatenate((images, image), axis=0)
                        # Add additional dimension corresponging to the number of batches (one in this case)
                images = np.expand_dims(images, axis=0)

                # Crop image to minimum box of non-background to save memory and avoid computations over the background
                inds_max = np.max(np.where(images[0, 0, :, :, :] > -3), axis=1)
                inds_min = np.min(np.where(images[0, 0, :, :, :] > -3), axis=1)
                images = images[:, :, inds_min[0]:inds_max[0] + 1, inds_min[1]:inds_max[1] + 1,
                         inds_min[2]:inds_max[2] + 1]

                # Pad image with zeros so that dimensions are multiples of 8. This is necesary due to the architecture
                pad = [(0, int(np.ceil(np.array(images.shape[i]) / 8.0) * 8 - np.array(images.shape[i])))
                       if i > 1 else (0, 0) for i in range(0, 5)]
                images = np.pad(images, pad, mode='constant', constant_values=-3)

                # Load labels and cut to min
                label_name = labelFilesTrain[data]

                label_data = nib.load(label_name)
                gt_label = label_data.get_data().astype('int')
                gt_label = gt_label[inds_min[0]:inds_max[0] + 1, inds_min[1]:inds_max[1] + 1,
                           inds_min[2]:inds_max[2] + 1]

                # Segment image
                pred = np.zeros((1,len(label_num),len(images[0,0]),len(images[0,0,0]), len(images[0,0,0,0])))
                for slice in range(len(images[0, 0, 0, 0])):
                    pred[:, :, :, :, slice] = segment_fn(images[:, :, :, :, slice])

                # Remove padding from prediction
                pred = np.squeeze(
                    pred[:pred.shape[0] - pad[0][1], :pred.shape[1] - pad[1][1], :pred.shape[2] - pad[2][1],
                    :pred.shape[3] - pad[3][1], :pred.shape[4] - pad[4][1]], axis=0)

                # Claculate Dice scores
                dice1 = calculate_dice(pred, gt_label, 1)
                train_dice1 += dice1
                train_images1 += 1
            train_dice1 /= train_images1


            # Write to log:
            f_log.write('{0}, {1}, {2}, {3}, {4}, {5}\n'.format( \
                epoch, confTrain['learningRate'], time.time() - start_time, train_loss, test_dice1, train_dice1))
            f_log.flush()

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(epoch, num_epochs, time.time() - start_time_epoch))
        print('  learning rate:\t\t{:.8f}'.format(float(learning_rate.get_value())))
        print("  training loss:\t\t{:.6f}".format(train_loss))
        print("  training Dice: \t\t{:.6f}".format(train_dice1))
        print("  testing Dice: \t\t{:.6f}".format(test_dice1))


        if lr_reduce:
            # Reduce the learning rate after each 50 epochs by 0.5
            if ((epoch % 50) == 0) & (epoch > 1):
                learning_rate.set_value(np.float32(learning_rate.get_value() * 0.5))
                confTrain['learningRate'] = confTrain['learningRate'] * 0.5

    # Close the logger
    f_log.close()
    print("Training took {:.3f}s in total.".format(time.time() - start_time))
