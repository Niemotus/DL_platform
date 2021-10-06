import theano.tensor as T
import lasagne
import lasagne.layers as L


def softmax_4dtensor(x):
    e_x = T.exp(x - x.max(axis=1, keepdims=True))
    e_x = e_x / e_x.sum(axis=1, keepdims=True)
    return e_x


def build_uResNet(image_var, shape=(None, 1, None, None), n_class=1):
    # Build fully-connected network for segmentation
    net = {}

    net['in']         = L.InputLayer(shape, image_var)
    net['conv1_1a']   = L.batch_norm(L.Conv2DLayer(net['in'],     filter_size=3, num_filters=32, pad='same',  nonlinearity=None))
    net['conv1_1b']   = L.batch_norm(L.Conv2DLayer(net['in'],     filter_size=1, num_filters=32, pad='valid', nonlinearity=None))
    net['fuse1_1']    = L.ElemwiseSumLayer([net['conv1_1a'],net['conv1_1b']])
    net['relu1']      = L.NonlinearityLayer(net['fuse1_1'],nonlinearity = lasagne.nonlinearities.rectify)
    net['pool1']      = L.MaxPool2DLayer(net['relu1'], pool_size=2)


    net['conv2_1a']   = L.batch_norm(L.Conv2DLayer(net['pool1'],   filter_size=3, num_filters=64, pad='same',  nonlinearity=None))
    net['conv2_1b']   = L.batch_norm(L.Conv2DLayer(net['pool1'],   filter_size=1, num_filters=64, pad='valid', nonlinearity=None))
    net['fuse2_1']    = L.ElemwiseSumLayer([net['conv2_1a'],net['conv2_1b']])
    net['relu2']      = L.NonlinearityLayer(net['fuse2_1'],nonlinearity = lasagne.nonlinearities.rectify)
    net['pool2']      = L.MaxPool2DLayer(net['relu2'], pool_size=2)


    net['conv3_1a']   = L.batch_norm(L.Conv2DLayer(net['pool2'],   filter_size=3, num_filters=128, pad='same',  nonlinearity=None))
    net['conv3_1b']   = L.batch_norm(L.Conv2DLayer(net['pool2'],   filter_size=1, num_filters=128, pad='valid', nonlinearity=None))
    net['fuse3_1']    = L.ElemwiseSumLayer([net['conv3_1a'],net['conv3_1b']])
    net['relu3']      = L.NonlinearityLayer(net['fuse3_1'],nonlinearity = lasagne.nonlinearities.rectify)
    net['pool3']      = L.MaxPool2DLayer(net['relu3'], pool_size=2)


    net['conv4_1a']   = L.batch_norm(L.Conv2DLayer(net['pool3'],   filter_size=3, num_filters=256, pad='same',  nonlinearity=None))
    net['conv4_1b']   = L.batch_norm(L.Conv2DLayer(net['pool3'],   filter_size=1, num_filters=256, pad='valid', nonlinearity=None))
    net['fuse4_1']    = L.ElemwiseSumLayer([net['conv4_1a'],net['conv4_1b']])
    net['relu4']      = L.NonlinearityLayer(net['fuse4_1'],nonlinearity = lasagne.nonlinearities.rectify)
    net['drop4']      = L.DropoutLayer(net['relu4'], p=0.5)
    net['deconv4']    = L.batch_norm(L.TransposedConv2DLayer(net['drop4'], num_filters = 128, filter_size=5, stride=2, nonlinearity=None))


    net['concat5']    = L.NonlinearityLayer(L.ElemwiseSumLayer([net['deconv4'],net['relu3']], cropping=[None, None, 'center', 'center']),nonlinearity = lasagne.nonlinearities.rectify)
    net['conv5_1a']   = L.batch_norm(L.Conv2DLayer(net['concat5'],   filter_size=3, num_filters=128, pad='same',  nonlinearity=None))
    net['conv5_1b']   = L.batch_norm(L.Conv2DLayer(net['concat5'],   filter_size=1, num_filters=128, pad='valid', nonlinearity=None))
    net['fuse5_1']    = L.ElemwiseSumLayer([net['conv5_1a'],net['conv5_1b']])
    net['relu5']      = L.NonlinearityLayer(net['fuse5_1'],nonlinearity = lasagne.nonlinearities.rectify)
    net['drop5']      = L.DropoutLayer(net['relu5'], p=0.5)
    net['deconv5']    = L.batch_norm(L.TransposedConv2DLayer(net['drop5'], num_filters = 64, filter_size=5, stride=2, nonlinearity=None))


    net['concat6']    = L.NonlinearityLayer(L.ElemwiseSumLayer([net['deconv5'],net['relu2']], cropping=[None, None, 'center', 'center']),nonlinearity = lasagne.nonlinearities.rectify)
    net['conv6_1a']   = L.batch_norm(L.Conv2DLayer(net['concat6'],   filter_size=3, num_filters=64, pad='same',  nonlinearity=None))
    net['conv6_1b']   = L.batch_norm(L.Conv2DLayer(net['concat6'],   filter_size=1, num_filters=64, pad='valid', nonlinearity=None))
    net['fuse6_1']    = L.ElemwiseSumLayer([net['conv6_1a'],net['conv6_1b']])
    net['relu6']      = L.NonlinearityLayer(net['fuse6_1'],nonlinearity = lasagne.nonlinearities.rectify)
    net['drop6']      = L.DropoutLayer(net['relu6'], p=0.5)
    net['deconv6']    = L.batch_norm(L.TransposedConv2DLayer(net['drop6'], num_filters = 32, filter_size=5, stride=2, nonlinearity=None))


    net['concat7']    = L.NonlinearityLayer(L.ElemwiseSumLayer([net['deconv6'],net['relu1']], cropping=[None, None, 'center', 'center']),nonlinearity = lasagne.nonlinearities.rectify)
    net['conv7_1a']   = L.batch_norm(L.Conv2DLayer(net['concat7'],   filter_size=3, num_filters=32, pad='same',  nonlinearity=None))
    net['conv7_1b']   = L.batch_norm(L.Conv2DLayer(net['concat7'],   filter_size=1, num_filters=32, pad='valid', nonlinearity=None))
    net['fuse7_1']    = L.ElemwiseSumLayer([net['conv7_1a'],net['conv7_1b']])
    net['relu7']      = L.NonlinearityLayer(net['fuse7_1'],nonlinearity = lasagne.nonlinearities.rectify)
    net['conv7_5']    = L.batch_norm(L.Conv2DLayer(net['relu7'],   filter_size=1, num_filters=n_class, pad='valid',  nonlinearity=None))
    net['output']     = L.Conv2DLayer(net['conv7_5'], num_filters=n_class, filter_size=1, nonlinearity=softmax_4dtensor)

    return net

