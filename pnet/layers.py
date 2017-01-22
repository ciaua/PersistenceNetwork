import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from lasagne import init
from lasagne import nonlinearities
from lasagne import layers
from lasagne.theano_extensions import padding
from lasagne.utils import as_tuple
from theano.tensor.nnet.neighbours import images2neibs as i2n
from theano.tensor.signal.pool import pool_2d
from theano.gradient import disconnected_grad as dg
from theano.ifelse import ifelse

floatX = theano.config.floatX


# modified from lasagne. Add 'strictsamex' for pad.
def conv_output_length(input_length, filter_size, stride, pad=0):
    """Helper function to compute the output size of a convolution operation

    This function computes the length along a single axis, which corresponds
    to a 1D convolution. It can also be used for convolutions with higher
    dimensionalities by using it individually for each axis.

    Parameters
    ----------
    input_length : int
        The size of the input.

    filter_size : int
        The size of the filter.

    stride : int
        The stride of the convolution operation.

    pad : int, 'full' or 'same' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.

        A single integer results in symmetric zero-padding of the given size on
        both borders.

        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.

        ``'same'`` pads with half the filter size on both sides (one less on
        the second side for an even filter size). When ``stride=1``, this
        results in an output size equal to the input size.

    Returns
    -------
    int
        The output size corresponding to the given convolution parameters.

    Raises
    ------
    RuntimeError
        When an invalid padding is specified, a `RuntimeError` is raised.
    """
    if input_length is None:
        return None
    if pad == 'valid':
        output_length = input_length - filter_size + 1
    elif pad == 'full':
        output_length = input_length + filter_size - 1
    elif pad == 'same':
        output_length = input_length
    elif pad == 'strictsamex':
        output_length = input_length
    elif isinstance(pad, int):
        output_length = input_length + 2 * pad - filter_size + 1
    else:
        raise ValueError('Invalid pad: {0}'.format(pad))

    # This is the integer arithmetic equivalent to
    # np.ceil(output_length / stride)
    output_length = (output_length + stride - 1) // stride

    return output_length


# modified from lasagne
def pool_output_length(input_length, pool_size, stride, pad, ignore_border):
    """
    Compute the output length of a pooling operator
    along a single dimension.

    Parameters
    ----------
    input_length : integer
        The length of the input in the pooling dimension
    pool_size : integer
        The length of the pooling region
    stride : integer
        The stride between successive pooling regions
    pad : integer
        The number of elements to be added to the input on each side.
    ignore_border: bool
        If ``True``, partial pooling regions will be ignored.
        Must be ``True`` if ``pad != 0``.

    Returns
    -------
    output_length
        * None if either input is None.
        * Computed length of the pooling operator otherwise.

    Notes
    -----
    When ``ignore_border == True``, this is given by the number of full
    pooling regions that fit in the padded input length,
    divided by the stride (rounding down).

    If ``ignore_border == False``, a single partial pooling region is
    appended if at least one input element would be left uncovered otherwise.
    """
    if input_length is None or pool_size is None:
        return None

    if pad == 'strictsame':
        output_length = input_length
    elif ignore_border:
        output_length = input_length + 2 * pad - pool_size + 1
        output_length = (output_length + stride - 1) // stride

    # output length calculation taken from:
    # https://github.com/Theano/Theano/blob/master/theano/tensor/signal/downsample.py
    else:
        assert pad == 0

        if stride >= pool_size:
            output_length = (input_length + stride - 1) // stride
        else:
            output_length = max(
                0, (input_length - pool_size + stride - 1) // stride) + 1

    return output_length


# add 'strictsamex' method for pad
class Pool2DXLayer(layers.Layer):
    """
    2D pooling layer

    Performs 2D mean or max-pooling over the two trailing axes
    of a 4D input tensor.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    pool_size : integer or iterable
        The length of the pooling region in each dimension.  If an integer, it
        is promoted to a square pooling region. If an iterable, it should have
        two elements.

    stride : integer, iterable or ``None``
        The strides between sucessive pooling regions in each dimension.
        If ``None`` then ``stride = pool_size``.

    pad : integer or iterable
        Number of elements to be added on each side of the input
        in each dimension. Each value must be less than
        the corresponding stride.

    ignore_border : bool
        If ``True``, partial pooling regions will be ignored.
        Must be ``True`` if ``pad != (0, 0)``.

    mode : {'max', 'average_inc_pad', 'average_exc_pad'}
        Pooling mode: max-pooling or mean-pooling including/excluding zeros
        from partially padded pooling regions. Default is 'max'.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    See Also
    --------
    MaxPool2DLayer : Shortcut for max pooling layer.

    Notes
    -----
    The value used to pad the input is chosen to be less than
    the minimum of the input, so that the output of each pooling region
    always corresponds to some element in the unpadded input region.

    Using ``ignore_border=False`` prevents Theano from using cuDNN for the
    operation, so it will fall back to a slower implementation.
    """

    def __init__(self, incoming, pool_size, stride=None, pad=(0, 0),
                 ignore_border=True, mode='max', **kwargs):
        super(Pool2DXLayer, self).__init__(incoming, **kwargs)

        self.pool_size = as_tuple(pool_size, 2)

        if stride is None:
            self.stride = self.pool_size
        else:
            self.stride = as_tuple(stride, 2)

        if pad == 'strictsamex':
            self.pad = pad
        else:
            self.pad = as_tuple(pad, 2)

        self.ignore_border = ignore_border
        self.mode = mode

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list

        if self.pad == 'strictsamex':
            output_shape[2] = pool_output_length(
                input_shape[2],
                pool_size=self.pool_size[0],
                stride=self.stride[0],
                pad='strictsame',
                ignore_border=self.ignore_border,
            )
            output_shape[3] = pool_output_length(
                input_shape[3],
                pool_size=self.pool_size[1],
                stride=self.stride[1],
                pad=0,
                ignore_border=self.ignore_border,
            )
        else:
            output_shape[2] = pool_output_length(
                input_shape[2],
                pool_size=self.pool_size[0],
                stride=self.stride[0],
                pad=self.pad[0],
                ignore_border=self.ignore_border,
            )

            output_shape[3] = pool_output_length(
                input_shape[3],
                pool_size=self.pool_size[1],
                stride=self.stride[1],
                pad=self.pad[1],
                ignore_border=self.ignore_border,
            )

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        if self.pad == 'strictsamex':
            assert(self.stride[0] == 1)
            kk = self.pool_size[0]
            ll = int(np.ceil(kk/2.))
            # rr = kk-ll
            # pad = (ll, 0)
            pad = [(ll, 0)]

            length = input.shape[2]

            self.ignore_border = True
            input = padding.pad(input, pad, batch_ndim=2)
            pad = (0, 0)
        else:
            pad = self.pad

        pooled = pool.pool_2d(input,
                              ds=self.pool_size,
                              st=self.stride,
                              ignore_border=self.ignore_border,
                              padding=pad,
                              mode=self.mode,
                              )

        if self.pad == 'strictsamex':
            pooled = pooled[:, :, :length or None, :]

        return pooled


# add 'strictsamex' method for pad
class MaxPool2DXLayer(Pool2DXLayer):
    """
    2D max-pooling layer

    Performs 2D max-pooling over the two trailing axes of a 4D input tensor.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    pool_size : integer or iterable
        The length of the pooling region in each dimension.  If an integer, it
        is promoted to a square pooling region. If an iterable, it should have
        two elements.

    stride : integer, iterable or ``None``
        The strides between sucessive pooling regions in each dimension.
        If ``None`` then ``stride = pool_size``.

    pad : integer or iterable
        Number of elements to be added on each side of the input
        in each dimension. Each value must be less than
        the corresponding stride.

    ignore_border : bool
        If ``True``, partial pooling regions will be ignored.
        Must be ``True`` if ``pad != (0, 0)``.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    Notes
    -----
    The value used to pad the input is chosen to be less than
    the minimum of the input, so that the output of each pooling region
    always corresponds to some element in the unpadded input region.

    Using ``ignore_border=False`` prevents Theano from using cuDNN for the
    operation, so it will fall back to a slower implementation.
    """

    def __init__(self, incoming, pool_size, stride=None, pad=(0, 0),
                 ignore_border=True, **kwargs):
        super(MaxPool2DXLayer, self).__init__(incoming,
                                              pool_size,
                                              stride,
                                              pad,
                                              ignore_border,
                                              mode='max',
                                              **kwargs)


# add 'strictsamex' method for pad
class Conv2DXLayer(layers.Layer):
    """
    lasagne.layers.Conv2DLayer(incoming, num_filters, filter_size,
    stride=(1, 1), pad=0, untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify,
    convolution=theano.tensor.nnet.conv2d, **kwargs)

    2D convolutional layer

    Performs a 2D convolution on its input and optionally adds a bias and
    applies an elementwise nonlinearity.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. The
        output of this layer should be a 4D tensor, with shape
        ``(batch_size, num_input_channels, input_rows, input_columns)``.

    num_filters : int
        The number of learnable convolutional filters this layer has.

    filter_size : int or iterable of int
        An integer or a 2-element tuple specifying the size of the filters.

    stride : int or iterable of int
        An integer or a 2-element tuple specifying the stride of the
        convolution operation.

    pad : int, iterable of int, 'full', 'same' or 'valid' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.

        A single integer results in symmetric zero-padding of the given size on
        all borders, a tuple of two integers allows different symmetric padding
        per dimension.

        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.

        ``'same'`` pads with half the filter size (rounded down) on both sides.
        When ``stride=1`` this results in an output size equal to the input
        size. Even filter size is not supported.

        ``'strictsamex'`` pads to the right of the third axis (x axis)
        to keep the same dim as input
        require stride=(1, 1)

        ``'valid'`` is an alias for ``0`` (no padding / a valid convolution).

        Note that ``'full'`` and ``'same'`` can be faster than equivalent
        integer values due to optimizations by Theano.

    untie_biases : bool (default: False)
        If ``False``, the layer will have a bias parameter for each channel,
        which is shared across all positions in this channel. As a result, the
        `b` attribute will be a vector (1D).

        If True, the layer will have separate bias parameters for each
        position in each channel. As a result, the `b` attribute will be a
        3D tensor.

    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a 4D tensor with shape
        ``(num_filters, num_input_channels, filter_rows, filter_columns)``.
        See :func:`lasagne.utils.create_param` for more information.

    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_filters,)`` if `untied_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, output_rows, output_columns)`` instead.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.

    convolution : callable
        The convolution implementation to use. Usually it should be fine to
        leave this at the default value.

    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.

    Attributes
    ----------
    W : Theano shared variable or expression
        Variable or expression representing the filter weights.

    b : Theano shared variable or expression
        Variable or expression representing the biases.

    Notes
    -----
    Theano's underlying convolution (:func:`theano.tensor.nnet.conv.conv2d`)
    only supports ``pad=0`` and ``pad='full'``. This layer emulates other modes
    by cropping a full convolution or explicitly padding the input with zeros.
    """
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 pad=0, untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 convolution=T.nnet.conv2d, **kwargs):
        super(Conv2DXLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_filters = num_filters
        self.filter_size = as_tuple(filter_size, 2)
        self.stride = as_tuple(stride, 2)
        self.untie_biases = untie_biases
        self.convolution = convolution

        if pad == 'same':
            if any(s % 2 == 0 for s in self.filter_size):
                raise NotImplementedError(
                    '`same` padding requires odd filter size.')
        if pad == 'strictsamex':
            if not (stride == 1 or stride == (1, 1)):
                raise NotImplementedError(
                    '`strictsamex` padding requires stride=(1, 1) or 1')

        if pad == 'valid':
            self.pad = (0, 0)
        elif pad in ('full', 'same', 'strictsamex'):
            self.pad = pad
        else:
            self.pad = as_tuple(pad, 2, int)

        self.W = self.add_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_filters, self.output_shape[2], self.
                                output_shape[3])
            else:
                biases_shape = (num_filters,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

    def get_W_shape(self):
        """Get the shape of the weight matrix `W`.

        Returns
        -------
        tuple of int
            The shape of the weight matrix.
        """
        num_input_channels = self.input_shape[1]
        return (self.num_filters, num_input_channels, self.filter_size[0],
                self.filter_size[1])

    def get_output_shape_for(self, input_shape):
        if self.pad == 'strictsamex':
            pad = ('strictsamex', 'valid')
        else:
            pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * 2

        output_rows = conv_output_length(input_shape[2],
                                         self.filter_size[0],
                                         self.stride[0],
                                         pad[0])

        output_columns = conv_output_length(input_shape[3],
                                            self.filter_size[1],
                                            self.stride[1],
                                            pad[1])

        return (input_shape[0], self.num_filters, output_rows, output_columns)

    def get_output_for(self, input, input_shape=None, **kwargs):
        # The optional input_shape argument is for when get_output_for is
        # called directly with a different shape than self.input_shape.
        if input_shape is None:
            input_shape = self.input_shape

        if self.stride == (1, 1) and self.pad == 'same':
            # simulate same convolution by cropping a full convolution
            conved = self.convolution(input, self.W, subsample=self.stride,
                                      input_shape=input_shape,
                                      # image_shape=input_shape,
                                      filter_shape=self.get_W_shape(),
                                      border_mode='full')
            crop_x = self.filter_size[0] // 2
            crop_y = self.filter_size[1] // 2
            conved = conved[:, :, crop_x:-crop_x or None,
                            crop_y:-crop_y or None]
        else:
            # no padding needed, or explicit padding of input needed
            if self.pad == 'full':
                border_mode = 'full'
                pad = [(0, 0), (0, 0)]
            elif self.pad == 'same':
                border_mode = 'valid'
                pad = [(self.filter_size[0] // 2,
                        self.filter_size[0] // 2),
                       (self.filter_size[1] // 2,
                        self.filter_size[1] // 2)]
            elif self.pad == 'strictsamex':
                border_mode = 'valid'
                kk = self.filter_size[0]-1
                rr = kk // 2
                ll = kk-rr
                pad = [(ll, rr),
                       (0, 0)]
            else:
                border_mode = 'valid'
                pad = [(self.pad[0], self.pad[0]), (self.pad[1], self.pad[1])]

            if pad != [(0, 0), (0, 0)]:
                input = padding.pad(input, pad, batch_ndim=2)
                input_shape = (input_shape[0], input_shape[1],
                               None if input_shape[2] is None else
                               input_shape[2] + pad[0][0] + pad[0][1],
                               None if input_shape[3] is None else
                               input_shape[3] + pad[1][0] + pad[1][1])
            conved = self.convolution(input, self.W, subsample=self.stride,
                                      input_shape=input_shape,
                                      # image_shape=input_shape,
                                      filter_shape=self.get_W_shape(),
                                      border_mode=border_mode)

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + self.b.dimshuffle('x', 0, 1, 2)
        else:
            activation = conved + self.b.dimshuffle('x', 0, 'x', 'x')

        return self.nonlinearity(activation)


# Persistence layers
class Persistence2DLayer(layers.Layer):
    """
    2D persistence-landscape layer

    Derive persistence landscapes for 2D cubical simplices
    Implement the algorithm in
    "Efficient Computation of Persistent Homology for Cubical data"

    Output shape:
        (batch_size, num_patches_0, num_patches_1,
         num_channels, num_pl_components, num_points)

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. The
        output of this layer should be a 4D tensor, with shape
        ``(batch_size, num_channels, input_height, input_width)``.

    num_pl_components : int
        The number of the sorted lambda functions used in persistence landscape

        *Do not* confuse this with the number of
        connected components of 0d homology classes

    num_points: int
        The number of points sampled from a lambda function

    value_range: tuple of float numbers
        A tuple in the form of (lower, upper)
        The points are sampled in the range [lower, upper]

    seg_size : int or iterable of int
        (length_0, length_1) specifies the segment size
        along the first axis and the second axis

        Corresponding to 'filter_size' in convolution
        An integer or a 1-element tuple specifying the size of the filters.

    seg_step : int or iterable of int
        (step_0, step_1) specifies the hop size of taking a segment
        along the first axis and the second axis

        Corresponding to 'stride' in convolution

    levelset_type : str
        'superlevel' or 'sublevel'

    add_extra_extrema_bd_pair : boolean
        True: the birth-death pair of (min, max) or (max, min) is included

    pad : int, iterable of int, 'full', 'same' or 'valid' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.

        An integer or a 1-element tuple results in symmetric zero-padding of
        the given size on both borders.

        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.

        ``'same'`` pads with half the filter size (rounded down) on both sides.
        When ``stride=1`` this results in an output size equal to the input
        size. Even filter size is not supported.

        ``'valid'`` is an alias for ``0`` (no padding / a valid convolution).

    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.

    """
    def __init__(self, incoming,
                 num_pl_components, num_points, value_range,
                 patch_size, patch_step=None, pl_dim=0,
                 levelset_type='superlevel',
                 add_extra_extrema_bd_pair=True,
                 pad=0,
                 **kwargs):
        super(Persistence2DLayer, self).__init__(incoming, **kwargs)

        # Note: I have to use (patch_step == patch_size)
        # because the images2neibs function can not handle the gradients in
        # other cases
        patch_step = patch_size
        assert(patch_step == patch_size)

        self.patch_size = as_tuple(patch_size, 2)
        self.patch_step = as_tuple(patch_step, 2)
        # self.stride = as_tuple(patch_step, 2)

        self.levelset_type = levelset_type
        self.add_extra_extrema_bd_pair = add_extra_extrema_bd_pair

        if pad == 'valid':
            self.pad = (0, 0)
        elif pad in ('full', 'same', 'strictsame'):
            self.pad = pad
        else:
            self.pad = as_tuple(pad, 2, int)

        self.num_pl_components = num_pl_components
        self.num_points = num_points
        self.value_range = value_range

        self.num_channels = self.input_shape[1]
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        self.pl_dim = pl_dim

    @classmethod
    def _signal2patch(cls, signal,
                      batch_size, num_channels, patch_size, patch_step,
                      do_padding=False):
        '''
        Slice raw input into patches

        signal : T.tensor4
            shape =
                (batch_size, num_channels, input_height, input_width)

        Convert the raw 2D input into smaller patches

        patch_size: tuple with two elements

        patch_step: tuple with two elements


        Return
        ------
        patches :
            shape =
                (batch_size,
                 num_patches_0, num_patches_1,
                 num_channels,
                 patch_size_0, patch_size_1)
        '''

        # batch_size = self.batch_size
        # num_channels = self.num_channels

        neib_shape = patch_size
        neib_step = patch_step

        input_height = signal.shape[2]
        input_width = signal.shape[3]

        patch_size_0 = np.array(patch_size[0], dtype='float32')
        patch_size_1 = np.array(patch_size[1], dtype='float32')

        if do_padding:
            # Padding
            num_patches_0 = T.ceil(
                input_height.astype('float32') / patch_size_0)
            len_rounded_0 = (patch_size_0*num_patches_0).astype('int32')

            num_patches_1 = T.ceil(
                input_width.astype('float32') / patch_size_1)
            len_rounded_1 = (patch_size_1*num_patches_1).astype('int32')

            signal = padding.pad(
                signal,
                [(0, len_rounded_0-input_height),
                 (0, len_rounded_1-input_width)], 0, 2).astype(floatX)
        else:
            num_patches_0 = T.floor(
                input_height.astype('float32') / patch_size_0)
            len_rounded_0 = (patch_size_0*num_patches_0).astype('int32')

            num_patches_1 = T.floor(
                input_width.astype('float32') / patch_size_1)
            len_rounded_1 = (patch_size_1*num_patches_1).astype('int32')

            signal = signal[:, :, :len_rounded_0, :len_rounded_1]

        patches = i2n(
            signal, neib_shape, neib_step=neib_step,
            # mode='ignore_borders'
            mode='valid'
        )

        '''
        num_patches_0 = utils.pool_output_length(
            input_height, patch_size[0], patch_step[0],
            pad=0,
            ignore_border=True
        )
        num_patches_1 = utils.pool_output_length(
            input_width, patch_size[1], patch_step[1],
            pad=0,
            ignore_border=True
        )
        '''

        num_patches_0 = num_patches_0.astype('int32')
        num_patches_1 = num_patches_1.astype('int32')

        patches = patches.reshape(
            (batch_size, num_channels,
             num_patches_0, num_patches_1,
             patch_size[0], patch_size[1]))
        patches = patches.dimshuffle((0, 2, 3, 1, 4, 5))

        return patches

    @classmethod
    def _build_cubemap(cls, patches, levelset_type='superlevel'):
        '''
        Implement CubeMap introduced in
        "Efficient Computation of Persistence Homology for Cubical Data"
        by Wagner et al.

        the upper half of "Algorithm 1"

        patches :
            shape =
                (batch_size,
                 num_patches_0, num_patches_1,
                 num_channels,
                 patch_size_0, patch_size_1)

        levelset_type : str
            'superlevel' for superlevel set
            'sublevel' for sublevel set


        Return
        ------

        cubemap :
            shape =
                (batch_size,
                 num_patches_0, num_patches_1,
                 num_channels,
                 2*patch_size_0+1, 2*patch_size_1+1)

        cubemap_ordering :
            shape =
                (batch_size,
                 num_patches_0, num_patches_1,
                 num_channels,
                 2*patch_size_0+1, 2*patch_size_1+1)

            Store the filtration indices of the cubes
            Indices are counted separately for different cube dimensions

        flatten_ordered_values_0d :
            the value at a given position corresponding to
            the value owned by the cube indicated by
            the index in cubemap_ordering
            That is,
            flatten_ordered_values_0d[......,
                                      cubemap_ordering[......, i]]
            is the value for that index.

            shape =
                (batch_size,
                 num_patches_0, num_patches_1,
                 num_channels,
                 num_cubes_0d)

        flatten_ordered_values_1d
            shape =
                (batch_size,
                 num_patches_0, num_patches_1,
                 num_channels,
                 num_cubes_1d)

        flatten_ordered_values_2d
            shape =
                (batch_size,
                 num_patches_0, num_patches_1,
                 num_channels,
                 num_cubes_2d)

        '''
        assert(levelset_type in ['superlevel', 'sublevel'])
        if levelset_type == 'superlevel':
            patches = -patches
        elif levelset_type == 'sublevel':
            pass

        shape = patches.shape
        cubemap_shape = (shape[0], shape[1], shape[2], shape[3],
                         shape[4]*2-1, shape[5]*2-1)
        cubemap = T.zeros(cubemap_shape)

        # cubemap for storing filtration indices
        cubemap_ordering = T.zeros(cubemap_shape)

        # cubemap_ordering = T.fill(cubemap_ordering, -np.inf)

        # Set 0D cubes (vertices)
        values_0d = patches
        cubemap = T.set_subtensor(cubemap[:, :, :, :, ::2, ::2], values_0d)

        flatten_values_0d = T.flatten(values_0d, outdim=5)
        flatten_idx_0d = T.argsort(flatten_values_0d, axis=-1)

        flatten_ordered_values_0d = T.sort(flatten_values_0d, axis=-1)

        flatten_ordering_0d = T.argsort(flatten_idx_0d, axis=-1)
        ordering_0d = flatten_ordering_0d.reshape(values_0d.shape)
        cubemap_ordering = T.set_subtensor(
            cubemap_ordering[:, :, :, :, ::2, ::2], ordering_0d)

        # flatten_shape_0d = flatten_idx_0d.shape

        # Set 1D cubes (edges)
        # #Vertical edges
        # values_1d_v = pool_2d(patches, (2, 1), stride=(1, 1),
        values_1d_v = pool_2d(patches, (2, 1), st=(1, 1),
                              mode='max', ignore_border=True)

        cubemap = T.set_subtensor(cubemap[:, :, :, :, 1::2, 0::2], values_1d_v)

        # #Horizontal edges
        # values_1d_h = pool_2d(patches, (1, 2), stride=(1, 1),
        values_1d_h = pool_2d(patches, (1, 2), st=(1, 1),
                              mode='max', ignore_border=True)

        cubemap = T.set_subtensor(cubemap[:, :, :, :, 0::2, 1::2], values_1d_h)

        # #Filtration index
        flatten_values_1d = T.concatenate(
            [T.flatten(values_1d_v, outdim=5),
             T.flatten(values_1d_h, outdim=5)], axis=4)
        flatten_ordered_values_1d = T.sort(flatten_values_1d, axis=-1)

        flatten_idx_1d = T.argsort(flatten_values_1d, axis=-1)
        flatten_ordering_1d = T.argsort(flatten_idx_1d, axis=-1)

        size_lastdim_v = T.flatten(values_1d_v, outdim=5).shape[-1]
        ordering_1d_v = flatten_ordering_1d[:, :, :, :, :size_lastdim_v]
        ordering_1d_h = flatten_ordering_1d[:, :, :, :, size_lastdim_v:]

        ordering_1d_v = ordering_1d_v.reshape(values_1d_v.shape)
        ordering_1d_h = ordering_1d_h.reshape(values_1d_h.shape)

        cubemap_ordering = T.set_subtensor(
            cubemap_ordering[:, :, :, :, 1::2, 0::2], ordering_1d_v)
        cubemap_ordering = T.set_subtensor(
            cubemap_ordering[:, :, :, :, 0::2, 1::2], ordering_1d_h)
        # flatten_shape_1d = flatten_idx_1d.shape

        # Set 2D cubes (faces)
        # values_2d = pool_2d(patches, (2, 2), stride=(1, 1),
        values_2d = pool_2d(patches, (2, 2), st=(1, 1),
                            mode='max', ignore_border=True)

        cubemap = T.set_subtensor(cubemap[:, :, :, :, 1::2, 1::2], values_2d)

        flatten_values_2d = T.flatten(values_2d, outdim=5)

        flatten_ordered_values_2d = T.sort(flatten_values_2d, axis=-1)

        flatten_idx_2d = T.argsort(flatten_values_2d, axis=-1)
        flatten_ordering_2d = T.argsort(flatten_idx_2d, axis=-1)

        ordering_2d = flatten_ordering_2d.reshape(values_2d.shape)
        cubemap_ordering = T.set_subtensor(
            cubemap_ordering[:, :, :, :, 1::2, 1::2], ordering_2d)
        # flatten_shape_2d = flatten_idx_2d.shape

        if levelset_type == 'superlevel':
            cubemap = -cubemap
            flatten_ordered_values_0d = -flatten_ordered_values_0d
            flatten_ordered_values_1d = -flatten_ordered_values_1d
            flatten_ordered_values_2d = -flatten_ordered_values_2d
        elif levelset_type == 'sublevel':
            pass

        return cubemap, cubemap_ordering, \
            flatten_ordered_values_0d, flatten_ordered_values_1d, \
            flatten_ordered_values_2d

    @classmethod
    def _make_ranges(cls, cubemap, batch_size, num_channels):
        num_patches_0 = cubemap.shape[1]
        num_patches_1 = cubemap.shape[2]
        map_size = cubemap.shape[4]*cubemap.shape[5]

        bs_range = T.tile(
            T.arange(batch_size, dtype='int32')[:, None, None, None, None],
            (1, num_patches_0, num_patches_1, num_channels, map_size)
        )
        np0_range = T.tile(
            T.arange(num_patches_0, dtype='int32')[None, :, None, None, None],
            (batch_size, 1, num_patches_1, num_channels, map_size)
        )
        np1_range = T.tile(
            T.arange(num_patches_1, dtype='int32')[None, None, :, None, None],
            (batch_size, num_patches_0, 1, num_channels, map_size)
        )
        nc_range = T.tile(
            T.arange(num_channels, dtype='int32')[None, None, None, :, None],
            (batch_size, num_patches_0, num_patches_1, 1, map_size)
        )

        return dg(bs_range), dg(np0_range), dg(np1_range), dg(nc_range)

    @classmethod
    def _make_ranges_for_bd(cls, birth_or_death, batch_size, num_channels):
        '''
        make ranges for birth/death
        '''
        bd = birth_or_death
        num_patches_0 = bd.shape[1]
        num_patches_1 = bd.shape[2]
        num_cubes = bd.shape[4]

        bs_range = T.tile(
            T.arange(batch_size, dtype='int32')[:, None, None, None, None],
            (1, num_patches_0, num_patches_1, num_channels, num_cubes)
        )
        np0_range = T.tile(
            T.arange(num_patches_0, dtype='int32')[None, :, None, None, None],
            (batch_size, 1, num_patches_1, num_channels, num_cubes)
        )
        np1_range = T.tile(
            T.arange(num_patches_1, dtype='int32')[None, None, :, None, None],
            (batch_size, num_patches_0, 1, num_channels, num_cubes)
        )
        nc_range = T.tile(
            T.arange(num_channels, dtype='int32')[None, None, None, :, None],
            (batch_size, num_patches_0, num_patches_1, 1, num_cubes)
        )

        return dg(bs_range), dg(np0_range), dg(np1_range), dg(nc_range)

    @classmethod
    def _build_boundary_matrix(cls, cubemap_ordering,
                               batch_size, num_channels):
        '''
        Construct boundary matrix introduced in
        "Efficient Computation of Persistence Homology for Cubical Data"
        by Wagner et al.

        the lower half of "Algorithm 1"

        cubemap_ordering :
            The ordering index of a given cube

            shape =
                (batch_size,
                 num_patches_0, num_patches_1,
                 num_channels,
                 2*patch_size_0+1, 2*patch_size_1+1)

        Return
        ------

        bm_0d1d, bm_1d2d :
            boundary matrices for 1d and 2d cubes

            shape =
                (batch_size,
                 num_patches_0, num_patches_1,
                 num_channels,
                 num_cubes_<i>d, num_cubes_<i+1>d)

        '''
        shape = cubemap_ordering.shape
        # print(shape[4].eval(), shape[5].eval())
        num_cubes_0d = (shape[4]+1)*(shape[5]+1)/4
        num_cubes_1d = (shape[4]*shape[5]-1)/2
        num_cubes_2d = (shape[4]-1)*(shape[5]-1)/4

        bm_0d1d_shape = (batch_size, shape[1], shape[2], num_channels,
                         num_cubes_0d, num_cubes_1d)
        bm_1d2d_shape = (batch_size, shape[1], shape[2], num_channels,
                         num_cubes_1d, num_cubes_2d)

        # Ranges
        bs_range, np0_range, np1_range, nc_range = cls._make_ranges(
            cubemap_ordering, batch_size, num_channels)

        # Boundary matrix 0d1d
        bm_0d1d = T.zeros(bm_0d1d_shape)

        # # Vertical
        idx_4_upper_v = cubemap_ordering[:, :, :, :, 0:-1:2, 0::2].flatten(
            ndim=5)
        idx_4_lower_v = cubemap_ordering[:, :, :, :, 2::2, 0::2].flatten(
            ndim=5)
        idx_5_v = cubemap_ordering[:, :, :, :, 1::2, 0::2].flatten(ndim=5)

        # # Horizontal
        idx_4_upper_h = cubemap_ordering[:, :, :, :, 0::2, 0:-1:2].flatten(
            ndim=5)
        idx_4_lower_h = cubemap_ordering[:, :, :, :, 0::2, 2::2].flatten(ndim=5)
        idx_5_h = cubemap_ordering[:, :, :, :, 0::2, 1::2].flatten(ndim=5)

        # # All. Every 1d cube has 2 0d cubes as the boundary
        idx_0 = bs_range[:, :, :, :, :num_cubes_1d*2]
        idx_1 = np0_range[:, :, :, :, :num_cubes_1d*2]
        idx_2 = np1_range[:, :, :, :, :num_cubes_1d*2]
        idx_3 = nc_range[:, :, :, :, :num_cubes_1d*2]

        idx_4 = T.concatenate([idx_4_upper_v, idx_4_lower_v,
                               idx_4_upper_h, idx_4_lower_h],
                              axis=4).astype('int32')
        idx_5 = T.concatenate([idx_5_v, idx_5_v, idx_5_h, idx_5_h],
                              axis=4).astype('int32')

        bm_0d1d = T.set_subtensor(
            bm_0d1d[idx_0, idx_1, idx_2, idx_3, idx_4, idx_5], 1)

        # If there is no element in the boundary matrix, return all zero bm
        temp_shape = (batch_size, shape[1], shape[2], num_channels, 1, 1)
        bm_0d1d = T.switch(T.gt(num_cubes_1d, 0), bm_0d1d, T.zeros(temp_shape))
        # bm_0d1d = ifelse(T.eq(num_cubes_1d, 0), T.zeros(temp_shape), bm_0d1d)

        # Boundary matrix 1d2d
        bm_1d2d = T.zeros(bm_1d2d_shape)

        # Index
        idx_5_oneside = cubemap_ordering[:, :, :, :, 1::2, 1::2].flatten(ndim=5)
        idx_4_upper = cubemap_ordering[:, :, :, :, 0:-1:2, 1::2].flatten(ndim=5)
        idx_4_lower = cubemap_ordering[:, :, :, :, 2::2, 1::2].flatten(ndim=5)
        idx_4_left = cubemap_ordering[:, :, :, :, 1::2, 0:-1:2].flatten(ndim=5)
        idx_4_right = cubemap_ordering[:, :, :, :, 1::2, 2::2].flatten(ndim=5)

        # # All
        idx_0 = bs_range[:, :, :, :, :num_cubes_2d*4]
        idx_1 = np0_range[:, :, :, :, :num_cubes_2d*4]
        idx_2 = np1_range[:, :, :, :, :num_cubes_2d*4]
        idx_3 = nc_range[:, :, :, :, :num_cubes_2d*4]

        idx_4 = T.concatenate([idx_4_upper, idx_4_lower,
                               idx_4_left, idx_4_right],
                              axis=4).astype('int32')
        idx_5 = T.concatenate([idx_5_oneside, idx_5_oneside,
                               idx_5_oneside, idx_5_oneside],
                              axis=4).astype('int32')

        bm_1d2d = T.set_subtensor(
            bm_1d2d[idx_0, idx_1, idx_2, idx_3, idx_4, idx_5], 1)

        # If there is no element in the boundary matrix, return all zero bm
        temp_shape = (batch_size, shape[1], shape[2], num_channels, 1, 1)
        bm_1d2d = T.switch(T.gt(num_cubes_2d, 0), bm_1d2d, T.zeros(temp_shape))
        # bm_1d2d = ifelse(T.eq(num_cubes_2d, 0), T.zeros(temp_shape), bm_1d2d)

        return bm_0d1d, bm_1d2d

    @classmethod
    def _do_reduction(cls, bm):
        '''
        Implement the reduction algorithm from
        "Topological Persistence and Simplification" by Edelsbrunner et al.

        bm :
            boundary matrix

        Return
        ------

        bm_reduced :
            Its shape is the same as bm.

        '''
        def _fn_for_reduce(current_row_idx, bm, mask_above):
            # sub_bm = bm[:, :, :, :, :, :current_column_idx+1]
            def _make_ranges_for_argmax(argmax):
                batch_size = argmax.shape[0]
                num_patches_0 = argmax.shape[1]
                num_patches_1 = argmax.shape[2]
                num_channels = argmax.shape[3]

                bs_range = T.tile(
                    T.arange(batch_size, dtype='int32')[
                        :, None, None, None],
                    (1, num_patches_0, num_patches_1, num_channels)
                )
                np0_range = T.tile(
                    T.arange(num_patches_0, dtype='int32')[
                        None, :, None, None],
                    (batch_size, 1, num_patches_1, num_channels)
                )
                np1_range = T.tile(
                    T.arange(num_patches_1, dtype='int32')[
                        None, None, :, None],
                    (batch_size, num_patches_0, 1, num_channels)
                )
                nc_range = T.tile(
                    T.arange(num_channels, dtype='int32')[
                        None, None, None, :],
                    (batch_size, num_patches_0, num_patches_1, 1)
                )

                return dg(bs_range), dg(np0_range), dg(np1_range), dg(nc_range)

            def __get_mask_max_raw(current_row, idx_max):
                zeros = T.zeros_like(current_row)
                mask_max_raw = T.set_subtensor(zeros[idx_max], 1)
                return mask_max_raw

            def _get_mask_max_raw(current_row):
                # zeros = T.zeros_like(current_row)
                # mask_max_raw = T.set_subtensor(zeros[idx_max], 1)

                max_value = current_row.max(axis=-1)

                temp = 1/(bm.shape[-1]).astype(floatX)
                tt = 0.5*T.arange(bm.shape[-1], dtype=floatX)[::-1]*temp

                tt = tt[None, None, None, None, :]
                mask_max_raw = T.eq(
                    (current_row+tt)-(current_row+tt).max(
                        axis=-1, keepdims=True),
                    0)

                return mask_max_raw, max_value

            # Current row
            current_row = bm[:, :, :, :, current_row_idx]*mask_above

            # Get max idx
            mask_max_raw, max_value = _get_mask_max_raw(current_row)

            # Mark max terms with 1
            mask_max = mask_max_raw*max_value[:, :, :, :, None]

            # Mark terms in the right of max term
            mask_right = current_row-mask_max_raw

            # zeros = T.zeros_like(current_row)
            # mask_max = T.set_subtensor(zeros[idx_max], 1) * \
            #     max_value[:, :, :, :, None]

            # Get the column with left-most one
            left_column = (mask_max[:, :, :, :, None, :]*bm).max(axis=5)
            # left_column = (mask_max[:, :, :, :, None, :]*bm).sum(axis=5)

            # Increment
            inc = left_column[:, :, :, :, :, None] *\
                mask_right[:, :, :, :, None, :]

            bm_new = bm+inc
            bm_new = T.mod(bm_new, 2)

            mask_above_new = mask_above-mask_max

            return bm_new, mask_above_new

        shape = bm.shape

        # A 5D tensor
        mask_above = T.ones(
            (shape[0], shape[1], shape[2], shape[3], shape[5])
        )

        num_rows = bm.shape[4]
        [bm_reduced, mask_above_final], _ = theano.reduce(
            fn=_fn_for_reduce,
            sequences=[T.arange(num_rows)],
            outputs_info=[bm, mask_above],
            go_backwards=True
        )

        return bm_reduced

    @classmethod
    def __do_reduction(cls, bm):
        '''
        Implement the reduction algorithm from
        "Topological Persistence and Simplification" by Edelsbrunner et al.

        bm :
            boundary matrix

        Return
        ------

        bm_reduced :
            Its shape is the same as bm.

        '''
        def _fn_for_reduce(current_row_idx, bm, mask_above):
            # sub_bm = bm[:, :, :, :, :, :current_column_idx+1]
            def _make_ranges_for_argmax(argmax):
                batch_size = argmax.shape[0]
                num_patches_0 = argmax.shape[1]
                num_patches_1 = argmax.shape[2]
                num_channels = argmax.shape[3]

                bs_range = T.tile(
                    T.arange(batch_size, dtype='int32')[
                        :, None, None, None],
                    (1, num_patches_0, num_patches_1, num_channels)
                )
                np0_range = T.tile(
                    T.arange(num_patches_0, dtype='int32')[
                        None, :, None, None],
                    (batch_size, 1, num_patches_1, num_channels)
                )
                np1_range = T.tile(
                    T.arange(num_patches_1, dtype='int32')[
                        None, None, :, None],
                    (batch_size, num_patches_0, 1, num_channels)
                )
                nc_range = T.tile(
                    T.arange(num_channels, dtype='int32')[
                        None, None, None, :],
                    (batch_size, num_patches_0, num_patches_1, 1)
                )

                return dg(bs_range), dg(np0_range), dg(np1_range), dg(nc_range)

            # Current row
            current_row = bm[:, :, :, :, current_row_idx]*mask_above

            # Get max idx
            argmax = current_row.argmax(axis=-1)
            idx_0, idx_1, idx_2, idx_3 = _make_ranges_for_argmax(argmax)
            idx_max = (idx_0, idx_1, idx_2, idx_3, argmax)

            max_value = current_row.max(axis=-1)

            # Mark terms in the right of max term
            mask_right = T.set_subtensor(current_row[idx_max], 0)

            # Mark max terms with 1
            mask_max = (current_row-mask_right)*max_value[:, :, :, :, None]

            # zeros = T.zeros_like(current_row)
            # mask_max = T.set_subtensor(zeros[idx_max], 1) * \
            #     max_value[:, :, :, :, None]

            # Get the column with left-most one
            left_column = (mask_max[:, :, :, :, None, :]*bm).max(axis=5)
            # left_column = (mask_max[:, :, :, :, None, :]*bm).sum(axis=5)

            # Increment
            inc = left_column[:, :, :, :, :, None] *\
                mask_right[:, :, :, :, None, :]

            bm_new = bm+inc
            bm_new = T.mod(bm_new, 2)

            mask_above_new = mask_above-mask_max

            return bm_new, mask_above_new

        shape = bm.shape

        # A 5D tensor
        mask_above = T.ones(
            (shape[0], shape[1], shape[2], shape[3], shape[5])
        )

        num_rows = bm.shape[4]
        [bm_reduced, mask_above_final], _ = theano.reduce(
            fn=_fn_for_reduce,
            sequences=[T.arange(num_rows)],
            outputs_info=[bm, mask_above],
            go_backwards=True
        )

        return bm_reduced

    @classmethod
    def _get_birth_death_pairs_in_index(cls, bm_reduced,
                                        num_cubes_higher_dim):
        '''
        Return
        ------

        birth_idx :
            index of the low-dim cubes that make some cubes birth

        death_idx :
            index of the high-dim cubes that make some cubes death


        '''
        # Get birth death index
        shape = bm_reduced.shape

        birth_idx = shape[4]-1-bm_reduced[:, :, :, :, ::-1].argmax(axis=4)

        death_idx = T.tile(
            T.arange(shape[5], dtype='int32')[
                None, None, None, None, :],
            (shape[0], shape[1], shape[2], shape[3], 1)
        )
        good = T.ge(bm_reduced.sum(axis=4), 1)

        birth_idx = good*birth_idx
        death_idx = good*death_idx

        return birth_idx, death_idx

    @classmethod
    def _get_birth_death_pairs_in_value(cls, bm_reduced,
                                        flatten_ordered_values_lowdim,
                                        flatten_ordered_values_highdim,
                                        batch_size, num_channels,
                                        num_cubes_higher_dim,
                                        add_max_min_pair=True):
        '''
        '''

        birth_idx, death_idx = \
            cls._get_birth_death_pairs_in_index(
                bm_reduced, num_cubes_higher_dim)

        # Ranges
        bs_range, np0_range, np1_range, nc_range = \
            cls._make_ranges_for_bd(birth_idx, batch_size, num_channels)

        birth_value = \
            flatten_ordered_values_lowdim[
                bs_range, np0_range, np1_range, nc_range, birth_idx]
        death_value = \
            flatten_ordered_values_highdim[
                bs_range, np0_range, np1_range, nc_range, death_idx]

        # If there is no element in the boundary matrix, return zeros
        shape = bm_reduced.shape
        # birth_value = T.switch(T.eq(shape[4]*shape[5], 1),
        #                        bm_reduced[..., 0], birth_value)
        # death_value = T.switch(T.eq(shape[4]*shape[5], 1),
        #                        bm_reduced[..., 0], death_value)

        temp = T.zeros_like(bm_reduced[:, :, :, :, :, 0])
        birth_value = ifelse(T.eq(shape[4]*shape[5], 1),
                             temp, birth_value)
        death_value = ifelse(T.eq(shape[4]*shape[5], 1),
                             temp, death_value)

        return birth_value, death_value

    @classmethod
    def _compute_persistence(cls, birth, death, levelset_type='superlevel'):
        '''
        levelset_type : str
            'superlevel' for superlevel set
            'sublevel' for sublevel set

        '''
        if levelset_type in ['superlevel']:
            persistence = birth-death
        elif levelset_type in ['sublevel']:
            persistence = death-birth

        return persistence

    @classmethod
    def _make_lambda_bd(cls, birth, death,
                        num_channels, num_points,
                        value_range, levelset_type):
        '''
        lambda_bd: 5-d tensor
            (batch_size, num_patches_0,num_patches_1, num_channels,
            num_bd_pairs, num_points)
        '''
        batch_size = birth.shape[0]
        num_patches_0 = birth.shape[1]
        num_patches_1 = birth.shape[2]

        value_range = np.array(value_range, dtype=floatX)

        num_bd_pairs = birth.shape[4]

        lambda_bd = T.zeros(
            (batch_size, num_patches_0, num_patches_1,
             num_channels, num_bd_pairs, num_points),
            dtype=floatX)

        lambda_bd += T.arange(num_points,
                              dtype=floatX)[None, None, None, None, None, :]
        lambda_bd /= np.array(num_points, dtype=floatX)
        lambda_bd = lambda_bd*(value_range[1]-value_range[0])+value_range[0]

        birth = birth.dimshuffle(0, 1, 2, 3, 4, 'x')
        death = death.dimshuffle(0, 1, 2, 3, 4, 'x')

        birth = T.addbroadcast(birth, 5)
        death = T.addbroadcast(death, 5)

        # lambda_bd = T.zeros_like(lambda_bd)

        bd_middle = (birth+death)/np.array(2, dtype=floatX)
        # print((lambda_bd-death_part).eval())

        if levelset_type in ['superlevel']:
            # Check "less than"
            lambda_bd = T.switch(T.lt(lambda_bd, bd_middle),
                                 lambda_bd-death, -lambda_bd+birth)
        elif levelset_type in ['sublevel']:
            lambda_bd = T.switch(T.lt(lambda_bd, bd_middle),
                                 lambda_bd-birth, -lambda_bd+death)
        lambda_bd = T.maximum(lambda_bd, np.array(0, dtype=floatX))

        return lambda_bd

    @classmethod
    def _lambda2pl(cls, lambda_bd, num_pl_components):
        '''
        lambda_bd to persistence diagram

        lambda_bd: 5-d tensor
            (batch_size, num_segments, num_channels,
             num_pl_components, num_points)

        pd: 5-d tensor
            persistence diagram
            (batch_size, num_segments, num_channels,
             num_pl_components, num_points)
        '''
        num_pad = T.maximum(num_pl_components-lambda_bd.shape[4], 0)

        lambda_bd = padding.pad(
            lambda_bd, [(0, num_pad), (0, 0)], 0, 4).astype(floatX)

        pl = lambda_bd.sort(axis=4)[:, :, :, :, ::-1, :]
        pl = pl[:, :, :, :, :num_pl_components]

        return pl

    @classmethod
    def _add_extrema_bd_pair(cls, birth_value, death_value,
                             flatten_ordered_values):
        '''
        Add the pair of min-max values as an additional birth-death pair
        '''
        birth_extra = flatten_ordered_values[:, :, :, :, 0:1]
        death_extra = flatten_ordered_values[:, :, :, :, -1:]
        birth = T.concatenate((birth_extra, birth_value), axis=4)
        death = T.concatenate((death_extra, death_value), axis=4)

        return birth, death

    @classmethod
    def get_birth_death(cls, signal,
                        patch_size, patch_step, levelset_type, dim,
                        add_extra_extrema_bd_pair=True):
        '''
        Build persistence landscape

        add_extra_extrema_bd_pair : boolean
            True: the birth-death pair of (min, max) or (max, min) is included

        return:
            birth, death
            shape = (batch_size, num_patches_0, num_patches_1, num_channels,
            num_bd_pairs)
        '''
        # Make input signal into smaller patches
        batch_size = signal.shape[0]
        num_channels = signal.shape[1]
        patches = cls._signal2patch(signal,
                                    batch_size, num_channels,
                                    patch_size, patch_step)

        # Build CubeMap
        cubemap, cubemap_ordering, \
            flatten_ordered_values_0d, flatten_ordered_values_1d, \
            flatten_ordered_values_2d = cls._build_cubemap(patches,
                                                           levelset_type)

        # Build boundary matrices
        bm_0d1d, bm_1d2d = cls._build_boundary_matrix(
            cubemap_ordering, batch_size, num_channels)
        bm_0d1d_reduced = cls._do_reduction(bm_0d1d)
        bm_1d2d_reduced = cls._do_reduction(bm_1d2d)

        # Get birth-death pairs
        num_cubes_2d = bm_1d2d.shape[-1]
        birth_0d, death_0d = cls._get_birth_death_pairs_in_value(
            bm_0d1d_reduced,
            flatten_ordered_values_0d,
            flatten_ordered_values_1d,
            batch_size, num_channels, num_cubes_2d
        )

        # Get birth-death pairs
        num_cubes_3d = 0
        birth_1d, death_1d = cls._get_birth_death_pairs_in_value(
            bm_1d2d_reduced,
            flatten_ordered_values_1d,
            flatten_ordered_values_2d,
            batch_size, num_channels, num_cubes_3d
        )

        # Add max-min pair
        if add_extra_extrema_bd_pair:
            birth_0d, death_0d = cls._add_extrema_bd_pair(
                birth_0d, death_0d, flatten_ordered_values_0d)
            birth_1d, death_1d = cls._add_extrema_bd_pair(
                birth_1d, death_1d, flatten_ordered_values_1d)

        if dim == 0:
            birth, death = birth_0d, death_0d
        elif dim == 1:
            birth, death = birth_1d, death_1d

        return birth, death

    @classmethod
    def get_birth_death_idx(cls, signal,
                            patch_size, patch_step, levelset_type, dim):
        '''
        Build persistence landscape
        '''
        # Make input signal into smaller patches
        batch_size = signal.shape[0]
        num_channels = signal.shape[1]
        patches = cls._signal2patch(signal,
                                    batch_size, num_channels,
                                    patch_size, patch_step)

        # Build CubeMap
        cubemap, cubemap_ordering, \
            flatten_ordered_values_0d, flatten_ordered_values_1d, \
            flatten_ordered_values_2d = cls._build_cubemap(patches,
                                                           levelset_type)

        # Build boundary matrices
        bm_0d1d, bm_1d2d = cls._build_boundary_matrix(
            cubemap_ordering, batch_size, num_channels)
        bm_0d1d_reduced = cls._do_reduction(bm_0d1d)
        bm_1d2d_reduced = cls._do_reduction(bm_1d2d)

        # Get birth-death pairs
        num_cubes_2d = bm_1d2d.shape[-1]
        birth_idx_0d, death_idx_0d = cls._get_birth_death_pairs_in_index(
            bm_0d1d_reduced, num_cubes_2d
        )

        # Get birth-death pairs
        num_cubes_3d = 0
        birth_idx_1d, death_idx_1d = cls._get_birth_death_pairs_in_index(
            bm_1d2d_reduced, num_cubes_3d
        )

        if dim == 0:
            birth_idx, death_idx = birth_idx_0d, death_idx_0d
        elif dim == 1:
            birth_idx, death_idx = birth_idx_1d, death_idx_1d

        return birth_idx, death_idx

    @classmethod
    def get_pl(cls, signal, num_pl_components, num_points, value_range,
               patch_size, patch_step, levelset_type, dim,
               add_extra_extrema_bd_pair):
        '''
        Build persistence landscape
        '''
        num_channels = signal.shape[1]
        birth, death = cls.get_birth_death(
            signal,
            patch_size, patch_step, levelset_type, dim,
            add_extra_extrema_bd_pair)

        # Make persistence landscape
        lambda_bd = cls._make_lambda_bd(
            birth, death,
            num_channels,
            num_points, value_range, levelset_type)
        pl = cls._lambda2pl(lambda_bd, num_pl_components)

        # output to the same form as convolution layer
        # pl_sym = pl_sym.flatten(ndim=3)

        return pl

    def get_output_shape_for(self, input_shape):
        '''
        shape:
        (batch_size, num_output_channels, output_length)
        '''
        if isinstance(self.pad, tuple):
            pad = self.pad
        else:
            pad = (self.pad, self.pad)

        if self.patch_size[0] is None:
            num_patches_0 = 1
            num_patches_1 = 1
        else:
            num_patches_0 = conv_output_length(
                input_shape[2],
                self.patch_size[0], self.patch_step[0], pad[0])
            num_patches_1 = conv_output_length(
                input_shape[3],
                self.patch_size[1], self.patch_step[1], pad[1])

        return (input_shape[0], num_patches_0, num_patches_1,
                self.num_channels, self.num_pl_components, self.num_points)

    def get_output_for(self, input, input_shape=None, **kwargs):
        # the optional input_shape argument is for when get_output_for is
        # called directly with a different shape than self.input_shape.

        # input = theano.printing.Print('Input to PL: ')(input)

        if input_shape is None:
            input_shape = self.input_shape

        self.batch_size = input.shape[0]

        if self.patch_size[0] is None:
            patch_size = input.shape[2:4]
            patch_step = input.shape[2:4]
        else:
            patch_size = self.patch_size
            patch_step = self.patch_step

        input_height = self.input_height
        input_width = self.input_width

        # no padding needed, or explicit padding of input needed
        if self.pad == 'full':
            border_mode = 'full'
            pad_h = (0, 0)
            pad_w = (0, 0)
        elif self.pad == 'same':
            border_mode = 'valid'
            pad_h = (input_height // 2, (patch_size[0]-1) // 2)
            pad_w = (input_width // 2, (patch_size[1]-1) // 2)
        else:
            border_mode = 'valid'
            pad_h = (self.pad[0], self.pad[0])
            pad_w = (self.pad[1], self.pad[1])

        if pad_h != (0, 0) or pad_w != (0, 0):
            input = padding.pad(input, [pad_h, pad_w], batch_ndim=2)
            input_shape = (input_shape[0], input_shape[1],
                           None if input_shape[2] is None else
                           input_shape[2] + pad_h[0] + pad_h[1],
                           None if input_shape[3] is None else
                           input_shape[3] + pad_w[0] + pad_w[1],
                           )

        if border_mode == 'full':
            input = padding.pad(input,
                                [patch_size[0]-1, patch_size[1]-1],
                                batch_ndim=2)
        output = self.get_pl(
            input, self.num_pl_components, self.num_points,
            self.value_range,
            patch_size, patch_step, self.levelset_type,
            self.pl_dim, self.add_extra_extrema_bd_pair)

        return output


class PersistenceFlatten2DLayer(Persistence2DLayer):
    """
    Flatten 2D persistence-landscape layer

    Similar to Persistence2Dlayer except that the output is flatten

    Input shape:
        (batch_size, num_patches_0, num_patches_1,
         num_channels, num_pl_components, num_points)

    Output shape:
        (batch_size, num_output_channels, num_patches_0, num_patches_1),
        where num_output_channels = num_channels*num_pl_components*num_points


    """
    def flatten(self, pl_sym):
        flatten_pl_sym = pl_sym.flatten(ndim=4).dimshuffle((0, 3, 1, 2))
        return flatten_pl_sym

    def get_output_shape_for(self, input_shape):
        '''
        shape:
        (batch_size, num_output_channels, output_length)
        '''
        shape_raw = super(PersistenceFlatten2DLayer,
                          self).get_output_shape_for(input_shape)

        return (shape_raw[0], shape_raw[3]*shape_raw[4]*shape_raw[5],
                shape_raw[1], shape_raw[2])

    def get_output_for(self, input, input_shape=None, **kwargs):
        output = super(PersistenceFlatten2DLayer, self).get_output_for(
            input, input_shape, **kwargs)

        flatten_output = self.flatten(output)
        return flatten_output
