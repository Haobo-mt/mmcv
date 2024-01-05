# Copyright (c) OpenMMLab. All rights reserved.
import os

import numpy as np
import pytest
import torch

from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE, IS_MUSA_AVAILABLE

_USING_PARROTS = True
try:
    from parrots.autograd import gradcheck
except ImportError:
    from torch.autograd import gradcheck

    _USING_PARROTS = False

cur_dir = os.path.dirname(os.path.abspath(__file__))

inputs = ([[[[0.88572276, 0.46422583], [0.97408265, 0.59547687],
             [0.030812204, 0.96236038], [0.75418317, 0.44058233],
             [0.33279222, 0.00084149837], [0.7069388, 0.23255438],
             [0.13547045, 0.81549376], [0.40174931, 0.36317211]],
            [[0.57444429, 0.15905505], [0.39897251, 0.25790238],
             [0.93282568, 0.18451685], [0.92526674, 0.18283755],
             [0.31664443, 0.59323865], [0.1957739, 0.42505842],
             [0.081158757, 0.81340349], [0.43456328, 0.30195212]],
            [[0.8198145, 0.05990988], [0.98062474, 0.34803438],
             [0.10412294, 0.37183142], [0.15021622, 0.038857818],
             [0.40985721, 0.42253625], [0.71150124, 0.59778064],
             [0.83851069, 0.15194464], [0.097513378, 0.74820143]],
            [[0.80680406, 0.49327564], [0.17821097, 0.12980539],
             [0.50657678, 0.14446253], [0.04178369, 0.53071898],
             [0.84983683, 0.3826949], [0.32193625, 0.91275406],
             [0.75628334, 0.52934098], [0.27994192, 0.3053292]]],
           [[[0.082397044, 0.4210068], [0.23563534, 0.7938987],
             [0.63669145, 0.69397897], [0.8844561, 0.97854084],
             [0.79027033, 0.60640401], [0.63528901, 0.72172403],
             [0.0097346902, 0.70800996], [0.87891227, 0.13674974]],
            [[0.74329448, 0.0243572], [0.82178867, 0.85750699],
             [0.7568835, 0.73146772], [0.5031184, 0.30479157],
             [0.28713053, 0.47414285], [0.4682079, 0.067471564],
             [0.48368263, 0.14590704], [0.25397325, 0.19946373]],
            [[0.4291026, 0.068739474], [0.7159555, 0.79903615],
             [0.76412082, 0.85348046], [0.081224024, 0.82264912],
             [0.97173303, 0.24291694], [0.48957139, 0.43488795],
             [0.67382395, 0.21889746], [0.36712623, 0.67127824]],
            [[0.12054044, 0.18096751], [0.86675781, 0.54755616],
             [0.68208277, 0.15164375], [0.79991871, 0.80811197],
             [0.85256428, 0.68253738], [0.185983, 0.95642138],
             [0.48102546, 0.28009653], [0.35726011, 0.58168036]]]])

shifts = [([[1, 0, 1, -2], [-2, 1, -1, 1]]), ([[2, 1, 2, -1], [-1, 2, 0, 2]])]

outputs = [([[[[0.0, 0.0], [0.0, 0.0], [0.030812, 0.96236], [0.75418, 0.44058],
               [0.0, 0.0], [0.0, 0.0], [0.83851, 0.15194], [0.097513, 0.7482]],
              [[0.88572, 0.46423], [0.97408, 0.59548], [0.93283, 0.18452],
               [0.92527, 0.18284], [0.33279, 0.0008415], [0.70694, 0.23255],
               [0.75628, 0.52934], [0.27994, 0.30533]],
              [[0.57444, 0.15906], [0.39897, 0.2579], [0.10412, 0.37183],
               [0.15022, 0.038858], [0.31664, 0.59324], [0.19577, 0.42506],
               [0.0, 0.0], [0.0, 0.0]],
              [[0.81981, 0.05991], [0.98062, 0.34803], [0.50658, 0.14446],
               [0.041784, 0.53072], [0.40986, 0.42254], [0.7115, 0.59778],
               [0.0, 0.0], [0.0, 0.0]]],
             [[[0.4291, 0.068739], [0.71596, 0.79904], [0.0, 0.0], [0.0, 0.0],
               [0.28713, 0.47414], [0.46821, 0.067472], [0.0, 0.0], [0.0,
                                                                     0.0]],
              [[0.12054, 0.18097], [0.86676, 0.54756], [0.63669, 0.69398],
               [0.88446, 0.97854], [0.97173, 0.24292], [0.48957, 0.43489],
               [0.0097347, 0.70801], [0.87891, 0.13675]],
              [[0.0, 0.0], [0.0, 0.0], [0.75688, 0.73147], [0.50312, 0.30479],
               [0.85256, 0.68254], [0.18598, 0.95642], [0.48368, 0.14591],
               [0.25397, 0.19946]],
              [[0.0, 0.0], [0.0, 0.0], [0.76412, 0.85348], [0.081224, 0.82265],
               [0.0, 0.0], [0.0, 0.0], [0.67382, 0.2189], [0.36713,
                                                           0.67128]]]]),
           ([[[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],
               [0.0, 0.0], [0.081159, 0.8134], [0.43456, 0.30195]],
              [[0.0, 0.0], [0.0, 0.0], [0.030812, 0.96236], [0.75418, 0.44058],
               [0.0, 0.0], [0.0, 0.0], [0.83851, 0.15194], [0.097513, 0.7482]],
              [[0.88572, 0.46423], [0.97408, 0.59548], [0.93283, 0.18452],
               [0.92527, 0.18284], [0.33279, 0.0008415], [0.70694, 0.23255],
               [0.75628, 0.52934], [0.27994, 0.30533]],
              [[0.57444, 0.15906], [0.39897, 0.2579], [0.10412, 0.37183],
               [0.15022, 0.038858], [0.31664, 0.59324], [0.19577, 0.42506],
               [0.0, 0.0], [0.0, 0.0]]],
             [[[0.74329, 0.024357], [0.82179, 0.85751], [0.0, 0.0], [0.0, 0.0],
               [0.79027, 0.6064], [0.63529, 0.72172], [0.0, 0.0], [0.0, 0.0]],
              [[0.4291, 0.068739], [0.71596, 0.79904], [0.0, 0.0], [0.0, 0.0],
               [0.28713, 0.47414], [0.46821, 0.067472], [0.0, 0.0], [0.0,
                                                                     0.0]],
              [[0.12054, 0.18097], [0.86676, 0.54756], [0.63669, 0.69398],
               [0.88446, 0.97854], [0.97173, 0.24292], [0.48957, 0.43489],
               [0.0097347, 0.70801], [0.87891, 0.13675]],
              [[0.0, 0.0], [0.0, 0.0], [0.75688, 0.73147], [0.50312, 0.30479],
               [0.85256, 0.68254], [0.18598, 0.95642], [0.48368, 0.14591],
               [0.25397, 0.19946]]]])]

grads = [
    [[[[0., 0.], [0., 0.], [1., 1.], [1., 1.], [0., 0.], [0., 0.], [1., 1.],
       [1., 1.]],
      [[1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.],
       [1., 1.]],
      [[1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [0., 0.],
       [0., 0.]],
      [[1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [0., 0.],
       [0., 0.]]],
     [[[1., 1.], [1., 1.], [0., 0.], [0., 0.], [1., 1.], [1., 1.], [0., 0.],
       [0., 0.]],
      [[1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.],
       [1., 1.]],
      [[0., 0.], [0., 0.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.],
       [1., 1.]],
      [[0., 0.], [0., 0.], [1., 1.], [1., 1.], [0., 0.], [0., 0.], [1., 1.],
       [1., 1.]]]],
    [[[[0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.], [1., 1.],
       [1., 1.]],
      [[0., 0.], [0., 0.], [1., 1.], [1., 1.], [0., 0.], [0., 0.], [1., 1.],
       [1., 1.]],
      [[1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.],
       [1., 1.]],
      [[1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [0., 0.],
       [0., 0.]]],
     [[[1., 1.], [1., 1.], [0., 0.], [0., 0.], [1., 1.], [1., 1.], [0., 0.],
       [0., 0.]],
      [[1., 1.], [1., 1.], [0., 0.], [0., 0.], [1., 1.], [1., 1.], [0., 0.],
       [0., 0.]],
      [[1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.],
       [1., 1.]],
      [[0., 0.], [0., 0.], [1., 1.], [1., 1.], [1., 1.], [1., 1.], [1., 1.],
       [1., 1.]]]]
]


def _test_tinshift_gradcheck(device, dtype):
    try:
        from mmcv.ops import tin_shift
    except ModuleNotFoundError:
        pytest.skip('TINShift op is not successfully compiled')

    if dtype == torch.half:
        pytest.skip('"add_cpu/sub_cpu" not implemented for Half')

    for shift in shifts:
        np_input = np.array(inputs)
        np_shift = np.array(shift)

        x = torch.tensor(
            np_input, dtype=dtype, device=device, requires_grad=True)
        shift = torch.tensor(np_shift, device=device).int()
        if torch.__version__ == 'parrots':
            gradcheck(tin_shift, (x, shift))
        else:
            gradcheck(tin_shift, (x, shift), atol=1, rtol=0.1)


def _test_tinshift_allclose(device, dtype):
    try:
        from mmcv.ops import tin_shift
    except ModuleNotFoundError:
        pytest.skip('TINShift op is not successfully compiled')

    for shift, output, grad in zip(shifts, outputs, grads):
        np_input = np.array(inputs)
        np_shift = np.array(shift)
        np_output = np.array(output)
        np_grad = np.array(grad)

        x = torch.tensor(
            np_input, dtype=dtype, device=device, requires_grad=True)
        shift = torch.tensor(np_shift, device=device).int()

        output = tin_shift(x, shift)
        output.backward(torch.ones_like(output))
        assert np.allclose(
            output.data.type(torch.float).cpu().numpy(), np_output, 1e-3)
        assert np.allclose(
            x.grad.data.type(torch.float).cpu().numpy(), np_grad, 1e-3)


def _test_tinshift_assert(device, dtype):
    try:
        from mmcv.ops import tin_shift
    except ModuleNotFoundError:
        pytest.skip('TINShift op is not successfully compiled')

    inputs = [
        torch.rand(2, 3, 4, 2),
        torch.rand(2, 3, 4, 2),
        torch.rand(1, 3, 4, 2)
    ]
    shifts = [torch.rand(2, 3), torch.rand(2, 5)]

    for x, shift in zip(inputs, shifts):
        x = x.to(device).type(dtype)
        shift = shift.to(device).type(dtype)

        # A ValueError should be raised if ops get inputs with wrong shapes.
        with pytest.raises(ValueError):
            tin_shift(x, shift)


@pytest.mark.parametrize('device', [
    pytest.param(
        'cuda',
        marks=pytest.mark.skipif(
            not IS_CUDA_AVAILABLE, reason='requires CUDA support')),
    pytest.param(
        'mlu',
        marks=pytest.mark.skipif(
            not IS_MLU_AVAILABLE, reason='requires MLU support')),
    pytest.param(
        'musa',
        marks=pytest.mark.skipif(
            not IS_MUSA_AVAILABLE, reason='requires MUSA support'))
])
@pytest.mark.parametrize('dtype', [
    torch.float,
    pytest.param(
        torch.double,
        marks=pytest.mark.skipif(
            IS_MLU_AVAILABLE or IS_MUSA_AVAILABLE,
            reason='MLU does not support for 64-bit floating point')),
    pytest.param(
        torch.half,
        marks=pytest.mark.skipif(
            IS_MUSA_AVAILABLE,
            reason='TODO haowen.han@mthreads.com: not supported yet')),
])
def test_tinshift(device, dtype):
    _test_tinshift_allclose(device=device, dtype=dtype)
    _test_tinshift_gradcheck(device=device, dtype=dtype)
    _test_tinshift_assert(device=device, dtype=dtype)
