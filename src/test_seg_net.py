import os
import theano
import theano.tensor as T
import lasagne
import numpy as np
import nibabel as nib
from network import build_uResNet
from helper_functions import categorical_dice
from Common import loadFilenames

if __name__ == '__main__':
    # PARAMETERS
    loss_name     = ''
    lr_reduce     = False
    debug         = False
    regularize    = False
    seed          = 10
    train         = 1
    epoch         = 300
    save_probability_maps = True
    np.random.seed(seed)


    base_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))

    # Load the traininf conf file (in this file the test set should be defined)
    confTrain = {}
    execfile(base_path + "/conf/training_dataset"+str(train)+".conf", confTrain)

    # Input channels
    print('Input channel lists:')
    for i in confTrain['channelsTraining']:
        print('\t'+i)

    # Load the training and testing image file names
    imageFilesTrain, labelFilesTrain, roiFilensTrain, trash = loadFilenames(
        confTrain['channelsTraining'], confTrain['gtLabelsTraining'] )
    imageFilesTest, labelFilesTest, roiFilensTest, trash = loadFilenames(
        confTrain['channelsValidation'],  confTrain['gtLabelsValidation'])

    # Get input channels from file name
    input_chan = [imageFilesTrain[i][0].split('/')[-1].split('.')[-3] for i in range(len(imageFilesTrain))]

    # Build the network
    net_name = 'uResNet2D'
    # Prepare theano variables
    image_var = T.tensor4('image')
    label_var = T.itensor4('label')
    weight_var= T.tensor4('weight')
    net = build_uResNet(image_var, shape=(None, len(confTrain['channelsTraining']), None, None),
                          n_class=len(confTrain['labels']))
    print("Build network:\t\t{:s}".format(net_name))
    print("L2 regularization: \t"+str(regularize))

    #get number of labels
    num_labels = len(confTrain["labels"])

    # Create a loss expression for testing. The crucial difference here is that we do a deterministic
    # forward pass through the network, disabling dropout layers.
    print("Create a function for testing...")
    test_prediction = lasagne.layers.get_output(net['output'], deterministic=True)

    # Compile a second function computing the testing loss and accuracy:
    segment_fn = theano.function([image_var], test_prediction)

    # Load saved model parameters and copy into the created net
    model_name = '{0}_DATASET{1}_ch-{2}_epoch{3:03d}{4}_seed{5}_L2-{6}.npz'.format(net_name, train, '-'.join(str(s) for s in input_chan), epoch, loss_name, seed, str(regularize))
    model_file = os.path.join(base_path, 'model', model_name)
    print('Test model:\t\t' + model_name)
    with np.load(model_file) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(net['output'], param_values)

    # Main testing loop
    test_dice = np.zeros((len(confTrain['labels'])))
    mean_dice = []
    for data in range(len(imageFilesTest[0])):
        print(str(data+1)+'\t'+imageFilesTest[0][data].split('/')[-2])

        # Load input channels
        for num_chan in range(len(imageFilesTest)):
            image_name = imageFilesTest[num_chan][data]
            print(image_name)
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
        images = images[:, :, inds_min[0]:inds_max[0]+1, inds_min[1]:inds_max[1]+1, inds_min[2]:inds_max[2]+1]

        # Pad image with zeros so that dimensions are multiples of 8. This is necesary due to the architecture
        pad = [(0, int(np.ceil(np.array(images.shape[i]) / 8.0) * 8 - np.array(images.shape[i])))
               if i > 1 else (0, 0) for i in range(0, 5)]
        images = np.pad(images, pad, mode='constant', constant_values=-3)

        # Load labels and cut to min
        label_name = labelFilesTest[data]
        label_data = nib.load(label_name)
        gt_label = label_data.get_data().astype('int')
        gt_label = gt_label[inds_min[0]:inds_max[0]+1, inds_min[1]:inds_max[1]+1, inds_min[2]:inds_max[2]+1]

        # Segment image
        pred = np.zeros((1, num_labels, len(images[0, 0]), len(images[0, 0, 0]), len(images[0, 0, 0, 0])))
        for slice in range(len(images[0, 0, 0, 0])):
            pred[:, :, :, :, slice] = segment_fn(images[:, :, :, :, slice])

        # Remove padding from prediction
        pred = pred[:pred.shape[0]-pad[0][1], :pred.shape[1]-pad[1][1], :pred.shape[2]-pad[2][1], :pred.shape[3]-pad[3][1], :pred.shape[4]-pad[4][1]]

        # Hard segmentation
        pred_hard = np.squeeze(np.argmax(pred.astype('float32'), axis=1), axis=0)

        # Calculate Dice
        for label_num in range(len(confTrain['labels'])):  # 14,15,33,34
            # dice = dice_fn(pred_hard.astype('float32'), gt_label.astype('float32'), label_num)
            dice = categorical_dice(pred_hard.astype('float32'), gt_label.astype('float32'), label_num)
            test_dice[label_num] += dice

        # Prepare hard segmentation output
        pred_hard_out = np.zeros(image_data.get_shape(), dtype='float32')
        pred_hard_out[inds_min[0]:inds_max[0]+1, inds_min[1]:inds_max[1]+1, inds_min[2]:inds_max[2]+1] = pred_hard

        # Prepare soft segmentation output
        pred_out = np.zeros((len(confTrain['labels']),)+image_data.get_shape(), dtype='float32')
        pred_out[:, inds_min[0]:inds_max[0]+1, inds_min[1]:inds_max[1]+1, inds_min[2]:inds_max[2]+1] = pred[0,:,:,:,:]

        # Save segmentation images to results folder
        if save_probability_maps:
            for label_num in range(len(confTrain['labels'])):
                nim = nib.Nifti1Image(pred_out[label_num, :, :, :].astype('float32'), image_data.affine)
                nib.save(nim, os.path.join(base_path + '/results/' +image_name.split('/')[-1].split('.')[-4] + '_Class' + str(label_num) + '.nii.gz'))
        nim = nib.Nifti1Image(pred_hard_out.astype('float32'), image_data.affine)
        nib.save(nim, os.path.join(base_path + '/results/' + image_name.split('/')[-1].split('.')[-4] + '_SEG' + '.nii.gz'))

        mean_dice.append(test_dice)
        print(test_dice)


    names = [imageFilesTest[0][i].split('/')[-1].split('.')[-4] for i in range(len(imageFilesTest[0]))]
    rows = np.array(names, dtype='|S20')[:, np.newaxis]
    np.savetxt(base_path + '/results/epoch' + str(epoch) + '_dice_scores.csv',
    np.hstack((rows, np.asarray(mean_dice))), delimiter=', ', fmt='%s')

    mean_dice.append(test_dice)
    print(test_dice)