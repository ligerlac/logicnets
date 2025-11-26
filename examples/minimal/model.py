#  Minimal LogicNets example - skip training, just demonstrate NN to HDL translation

import torch
import torch.nn as nn

from brevitas.core.quant import QuantType
from brevitas.core.scaling import ScalingImplType
from brevitas.nn import QuantHardTanh, QuantReLU

from logicnets.quant import QuantBrevitasActivation
from logicnets.nn import SparseLinearNeq, ScalarBiasScale, RandomFixedSparsityMask2D


class MinimalNeqModel(nn.Module):
    """
    Minimal neural network for demonstrating LogicNets NN-to-HDL translation.
    Architecture: 4 inputs -> 3 hidden neurons -> 2 outputs
    Uses 2-bit quantization and low fan-in for minimal FPGA resources.
    """
    def __init__(self):
        super(MinimalNeqModel, self).__init__()

        # Network architecture
        input_length = 4
        hidden_size = 3
        output_length = 2

        # Quantization settings (2-bit for minimal LUT usage)
        input_bitwidth = 2
        hidden_bitwidth = 2
        output_bitwidth = 2

        # Fan-in settings (sparse connectivity)
        input_fanin = 2   # Each hidden neuron connects to 2 inputs
        output_fanin = 2  # Each output connects to 2 hidden neurons

        layer_list = []

        # Layer 1: Input -> Hidden (4 -> 3)
        bn_in = nn.BatchNorm1d(input_length)
        input_bias = ScalarBiasScale(scale=False, bias_init=-0.25)
        input_quant = QuantBrevitasActivation(
            QuantHardTanh(input_bitwidth, max_val=1., narrow_range=False,
                         quant_type=QuantType.INT,
                         scaling_impl_type=ScalingImplType.PARAMETER),
            pre_transforms=[bn_in, input_bias]
        )

        bn1 = nn.BatchNorm1d(hidden_size)
        hidden_quant = QuantBrevitasActivation(
            QuantReLU(bit_width=hidden_bitwidth, max_val=1.61,
                     quant_type=QuantType.INT,
                     scaling_impl_type=ScalingImplType.PARAMETER),
            pre_transforms=[bn1]
        )

        mask1 = RandomFixedSparsityMask2D(input_length, hidden_size, fan_in=input_fanin)
        layer1 = SparseLinearNeq(input_length, hidden_size,
                                input_quant=input_quant,
                                output_quant=hidden_quant,
                                sparse_linear_kws={'mask': mask1})
        layer_list.append(layer1)

        # Layer 2: Hidden -> Output (3 -> 2)
        bn2 = nn.BatchNorm1d(output_length)
        output_bias_scale = ScalarBiasScale(bias_init=0.33)
        output_quant = QuantBrevitasActivation(
            QuantHardTanh(bit_width=output_bitwidth, max_val=1.33,
                         narrow_range=False,
                         quant_type=QuantType.INT,
                         scaling_impl_type=ScalingImplType.PARAMETER),
            pre_transforms=[bn2],
            post_transforms=[output_bias_scale]
        )

        mask2 = RandomFixedSparsityMask2D(hidden_size, output_length, fan_in=output_fanin)
        layer2 = SparseLinearNeq(hidden_size, output_length,
                                input_quant=layer1.output_quant,
                                output_quant=output_quant,
                                sparse_linear_kws={'mask': mask2},
                                apply_input_quant=False)
        layer_list.append(layer2)

        self.module_list = nn.ModuleList(layer_list)

    def forward(self, x):
        for layer in self.module_list:
            x = layer(x)
        return x


class MinimalLutModel(nn.Module):
    """
    LUT-based version of MinimalNeqModel for Verilog generation.
    Identical architecture but uses LUT inference after truth table generation.
    """
    def __init__(self):
        super(MinimalLutModel, self).__init__()

        # Identical architecture to MinimalNeqModel
        input_length = 4
        hidden_size = 3
        output_length = 2

        input_bitwidth = 2
        hidden_bitwidth = 2
        output_bitwidth = 2

        input_fanin = 2
        output_fanin = 2

        layer_list = []

        # Layer 1: Input -> Hidden (4 -> 3)
        bn_in = nn.BatchNorm1d(input_length)
        input_bias = ScalarBiasScale(scale=False, bias_init=-0.25)
        input_quant = QuantBrevitasActivation(
            QuantHardTanh(input_bitwidth, max_val=1., narrow_range=False,
                         quant_type=QuantType.INT,
                         scaling_impl_type=ScalingImplType.PARAMETER),
            pre_transforms=[bn_in, input_bias]
        )

        bn1 = nn.BatchNorm1d(hidden_size)
        hidden_quant = QuantBrevitasActivation(
            QuantReLU(bit_width=hidden_bitwidth, max_val=1.61,
                     quant_type=QuantType.INT,
                     scaling_impl_type=ScalingImplType.PARAMETER),
            pre_transforms=[bn1]
        )

        mask1 = RandomFixedSparsityMask2D(input_length, hidden_size, fan_in=input_fanin)
        layer1 = SparseLinearNeq(input_length, hidden_size,
                                input_quant=input_quant,
                                output_quant=hidden_quant,
                                sparse_linear_kws={'mask': mask1})
        layer_list.append(layer1)

        # Layer 2: Hidden -> Output (3 -> 2)
        bn2 = nn.BatchNorm1d(output_length)
        output_bias_scale = ScalarBiasScale(bias_init=0.33)
        output_quant = QuantBrevitasActivation(
            QuantHardTanh(bit_width=output_bitwidth, max_val=1.33,
                         narrow_range=False,
                         quant_type=QuantType.INT,
                         scaling_impl_type=ScalingImplType.PARAMETER),
            pre_transforms=[bn2],
            post_transforms=[output_bias_scale]
        )

        mask2 = RandomFixedSparsityMask2D(hidden_size, output_length, fan_in=output_fanin)
        layer2 = SparseLinearNeq(hidden_size, output_length,
                                input_quant=layer1.output_quant,
                                output_quant=output_quant,
                                sparse_linear_kws={'mask': mask2},
                                apply_input_quant=False)
        layer_list.append(layer2)

        self.module_list = nn.ModuleList(layer_list)

    def forward(self, x):
        for layer in self.module_list:
            x = layer(x)
        return x
