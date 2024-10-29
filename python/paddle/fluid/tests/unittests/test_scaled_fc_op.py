#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from paddle.static import Program, program_guard
import paddle.fluid as fluid

np.set_printoptions(suppress=True)

class TestScaledFcOp(OpTest):

    def setUp(self):

        self.__class__.op_type = "scaled_fc"
        self.dtype = np.float32
        self.python_api = fluid.contrib.layers.scaled_fc
        self.init_kernel_type()
        self.place = core.XPUPlace(0)
        # self.place = core.CUDAPlace(0)
        # self.place = paddle.fluid.core_avx.CPUPlace()
        input = np.random.random((4, 25)).astype(self.dtype)
        w = np.random.random((25,25)).astype(self.dtype)
        bias = np.random.random((25, 1)).astype(self.dtype)
        self.inputs = {'Input': input, 'W': w, 'Bias': bias}
        self.outputs = {
            'Out': self.inputs['Input'] #bad data
        }
        self.attrs = {
            'input_scale_factor':  0.0078125,
            'bias_scale_factor':  0.0078125,
            'grad_scale_factor':  0.0078125
        }

    def init_kernel_type(self):
        pass

    def test_check_output(self):
          self.check_output_with_place(self.place, check_eager=False)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place,
            ["Input", "W", "Bias"],
            "Out",
            max_relative_error=10)


class TestScaledFcApi(unittest.TestCase):
    def setUp(self):
      self.place = core.XPUPlace(0)
      # self.place = core.CUDAPlace(0)
      # self.place = paddle.fluid.core_avx.CPUPlace()
      self.input_data = np.random.random([2, 8]).astype("float32")
      self.w_data = np.random.random([8, 8]).astype("float32")
      self.bias_data = np.random.random([8, 1]).astype("float32")
      # self.w_data = np.random.normal(loc=0.0, scale=1.0 * 0.00614, size=(8, 8)).astype("float32")
      # self.bias_data = np.random.normal(loc=0.0, scale=1.0 * 0.00614, size=(8,1)).astype("float32")
      self.input_scale_factor = 0.0078125
      self.bias_scale_factor = 0.0078125
      self.wsize = []
      self.wsize.append(self.w_data.shape[0])
      self.wsize.append(self.w_data.shape[1])

      self.bsize = []
      self.bsize.append(self.bias_data.shape[0])
      self.bsize.append(self.bias_data.shape[1])
      self.w = paddle.ParamAttr(learning_rate=1.0,
                          name ="scaled_fc.w_0",
                          initializer=paddle.nn.initializer.Normal(mean=0.0, std=1.0 * 0.00614))
      self.bias = paddle.ParamAttr(learning_rate=1.0,
                          name = "scaled_fc.b_0",
                          initializer=paddle.nn.initializer.Normal(mean = 0.0, std = 1.0 * 0.00614))

    def _executed_api(self, input, wsize, w, bsize, bias, input_scale_factor, bias_scale_factor):
        return fluid.contrib.layers.scaled_fc(input,
                                              wsize,
                                              w,
                                              bsize,
                                              bias,
                                              input_scale_factor,
                                              bias_scale_factor)


    def test_api_dygraph(self):
        paddle.disable_static(self.place)
        input = paddle.to_tensor(self.input_data)
        out = self._executed_api(input, self.wsize, self.w, self.bsize, self.bias, self.input_scale_factor, self.bias_scale_factor)
        print('dygraph_out:',out)
        # self.assertEqual(np.array_equal(out.numpy(), input * 2.0 + 3.0), True)
        paddle.enable_static()


    def test_api_static(self):
        paddle.enable_static()
        np.random.seed(0)
        self.input_data = np.random.random([4, 8]).astype("float32")
        main_prog = Program()
        with program_guard(main_prog, Program()):
            input = paddle.fluid.data(name="input", shape=[4, 8], dtype="float32")
            out = self._executed_api(input, self.wsize, self.w, self.bsize, self.bias, self.input_scale_factor, self.bias_scale_factor)
        exe = paddle.static.Executor(self.place)
        res = exe.run(main_prog, feed={"input": self.input_data,
                                        "scaled_fc.w_0":  self.w_data,
                                        "scaled_fc.b_0":  self.bias_data}, fetch_list=[out])
        print('static_out:',res)
        # self.assertEqual(np.array_equal(out[0], input * 2.0 + 3.0), True)


if __name__ == "__main__":
    np.random.seed(0)
    paddle.seed(0)
    unittest.main()
