# -*- coding: utf-8 -*-
"""核心量化逻辑测试 - 双分支共享
Copyright (C) 2026 jakerdliu (Email: koball@263.net)
GitHub: https://github.com/jakerdliu/OpenTrit-CHN.git
"""
import unittest
import numpy as np
from opentrit_chn import calculate_entropy, apply_ternary_quantization

class TestCoreQuantization(unittest.TestCase):
    def test_calculate_entropy(self):
        """测试信息熵计算"""
        tensor = np.array([-1, 0, 1, 0, 1, -1])
        entropy = calculate_entropy(tensor)
        self.assertGreater(entropy, 0)
        self.assertLess(entropy, 2)

    def test_ternary_quantization(self):
        """测试三值量化"""
        tensor = np.array([-0.2, -0.1, 0, 0.1, 0.2])
        ternary = apply_ternary_quantization(tensor, threshold=0.15)
        self.assertEqual(list(ternary), [-1, 0, 0, 0, 1])

if __name__ == "__main__":
    unittest.main()