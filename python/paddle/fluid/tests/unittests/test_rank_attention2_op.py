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
import random

class TestRankAttentionOpComplex(OpTest):

    def config(self):
        self.ins_num = 100
        self.x_feat = 10
        self.y_feat = 15
        self.max_rank = 3
        self.dtype = "float32"
        self.place = core.XPUPlace(0)
        if core.is_compiled_with_cuda():
          self.place = core.CUDAPlace(0)

    def setUp(self):
        self.__class__.op_type = "rank_attention2"
        self.config()
        input = np.random.random((self.ins_num, self.x_feat)).astype(self.dtype)
        rank_offset = np.random.randint(0, 5, (self.ins_num, 7)).astype("int32")
        rank_para_shape = [
            self.max_rank * self.max_rank * self.x_feat, self.y_feat
        ]
        rank_para = np.random.random(rank_para_shape).astype(self.dtype)
        self.inputs = {
            "X": input,
            "RankOffset": np.array(rank_offset).astype("int32"),
            "RankParam": rank_para
        }
        self.attrs = {'MaxRank': self.max_rank, 'MaxSize': self.pv_num * 7}
        self.outputs = {
            "Out": rank_offset #bad data
        }

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ["RankParam"], "Out")



class TestRankAttention2ApiStatic(unittest.TestCase):

    def _executed_api(self, input, rank_offset):
        return fluid.contrib.layers.rank_attention2(input,
                                    rank_offset,
                                    [input.shape[1] * 3 * 3, 48],
                                    rank_param_attr=fluid.ParamAttr(
                                        learning_rate=1.0,
                                        name="rank_attention2.w_0",
                                        initializer=fluid.initializer.Xavier(uniform=False)
                                        )
                                    )


    def test_api(self):
        paddle.enable_static()
        input_data = np.random.random([5, 25]).astype("float32")
        w_data = np.random.random([25*3*3,48]).astype("float32")
        rank_offset_data = np.random.randint(0, 5, (5, 7)).astype("int32")
        main_prog = Program()
        with program_guard(main_prog, Program()):
            input = paddle.static.data(name="input", shape=[5, 25], dtype="float32")
            rank_offset = paddle.static.data(name="rank_offset",shape=[5, 7],dtype="int32")
            out = self._executed_api(input, rank_offset)

        place = paddle.device.get_device()
        # place = paddle.fluid.core_avx.CPUPlace()
        exe = paddle.static.Executor(place)
        out = exe.run(main_prog, feed={"input": input_data, "rank_attention2.w_0": w_data, "rank_offset": rank_offset_data}, fetch_list=[out])
        print('out:',out)
        # self.assertEqual(np.array_equal(out[0], input * 2.0 + 3.0), True)

if __name__ == "__main__":
    np.random.seed(102)
    unittest.main()
