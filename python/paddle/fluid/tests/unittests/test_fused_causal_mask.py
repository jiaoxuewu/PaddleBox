#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np
import random
import itertools
import paddle
import paddle.fluid as fluid


def get_sequence_causal_mask_paddle(seq_info: paddle.Tensor) -> paddle.Tensor:
    """
    Generates a causal mask (including diagonal) for sequences using PaddlePaddle.

    Args:
        seq_info (paddle.Tensor): Tensor with exclusive end indices of each sequence. Shape: [num_sequences + 1].

    Returns:
        paddle.Tensor: A lower triangular boolean mask where each sequence attends to itself and preceding elements.

    Example:
        >>> seq_info = paddle.to_tensor([0, 3, 5])
        >>> print(get_sequence_causal_mask_paddle(seq_info))
        array([[ True, False, False, False, False],
               [ True,  True, False, False, False],
               [ True,  True,  True, False, False],
               [False, False, False,  True, False],
               [False, False, False,  True,  True]])
    """
    lengths = seq_info[1:] - seq_info[:-1]
    lengths = paddle.reshape(lengths, [-1])
    indices = paddle.cumsum(paddle.ones_like(lengths)) - 1
    result = paddle.repeat_interleave(indices, lengths)
    a = result.reshape([1, -1]) - result.reshape([-1, 1])
    return paddle.tril(paddle.cast(a == 0, "int64"))


class TestFusedCausalMask(unittest.TestCase):
    def setUp(self):
        self.rtol = 1e-5
        # FIXME(limin29): Because there is a problem with the test precision
        #  on A100, atol is temporarily set to 1e-2, and it will be
        #  changed back after the precision problem is solved.
        self.atol = 1e-2
        # make sure local development precision
        if "V100" in paddle.device.cuda.get_device_name():
            self.atol = 1e-4

        random.seed(100)

    def generate_data(self):
        seq_info = list(random.randint(1, 512) for i in range(40))
        seq_info = [0] + list(itertools.accumulate(seq_info))
        # seq_info = [0, 4, 10]
        x_data = np.array(seq_info, dtype="int32")
        return x_data

    def test_static_mode(self):
        # 定义静态图
        paddle.enable_static()

        x = fluid.layers.data(name="x", shape=[-1], dtype="int32")
        output_paddle = get_sequence_causal_mask_paddle(x)
        output_fused = fluid.contrib.layers.fused_causal_mask(x)

        # 指定使用GPU进行计算
        paddle.set_device("gpu:0")
        exe = fluid.Executor()
        exe.run(fluid.default_startup_program())
        for i in range(10):
            x_data = self.generate_data()
            output = exe.run(
                program=fluid.default_main_program(),
                feed={"x": x_data},
                fetch_list=[output_paddle, output_fused],
            )
            np.testing.assert_allclose(
                output[0], output[1], rtol=self.rtol, atol=self.atol
            )


if __name__ == "__main__":
    unittest.main()
