# Minimal LogicNets Example

This is a minimal example demonstrating how LogicNets translates a trained neural network into Verilog HDL for FPGA deployment.

## Overview

This example **skips training entirely** and focuses on understanding the NN-to-HDL translation pipeline. It uses a tiny neural network with randomly initialized weights to demonstrate:

1. Neural network quantization and sparsification
2. Conversion to LUT-based representation
3. Verilog HDL generation
4. FPGA resource estimation

## Network Architecture

**Minimal 2-layer network:**
- **Input layer**: 4 inputs, 2-bit quantization
- **Hidden layer**: 3 neurons, 2-bit quantization, fan-in=2
- **Output layer**: 2 outputs, 2-bit quantization, fan-in=2

**Total parameters**: ~9 sparse connections (vs 4×3 + 3×2 = 18 for fully connected)

## Translation Pipeline

```
Random PyTorch NN (float32)
    ↓
Quantized with Brevitas (2-bit INT)
    ↓
Sparsified (fan-in = 2)
    ↓
Generate Truth Tables (NEQ → LUT)
    ↓
Verilog Generation (ROM-based LUTs)
    ↓
Vivado Synthesis
    ↓
FPGA Resource Report
```

## Usage

### Quick Start (Verilog only, no synthesis)

```bash
cd examples/minimal
python convert_to_verilog.py --skip-synthesis
```

This will:
- Create a minimal NN with random weights
- Convert it to LUT-based representation
- Generate Verilog in `./output/`
- Show estimated LUT cost

### Full Pipeline (with Vivado synthesis)

```bash
python convert_to_verilog.py --fpga-part xc7a35tcpg236-1
```

**Requirements**: Vivado and [oh-my-xilinx](https://github.com/Xilinx/oh-my-xilinx) must be installed.

### Options

```bash
python convert_to_verilog.py --help
```

- `--output-dir DIR`: Where to save Verilog (default: `./output`)
- `--clock-period NS`: Target clock period in ns (default: 10.0)
- `--fpga-part PART`: FPGA part number (default: `xc7a35tcpg236-1`)
- `--skip-synthesis`: Only generate Verilog, skip Vivado synthesis
- `--add-registers`: Add pipeline registers between layers (improves fmax, adds latency)

## Output Files

After running, you'll find in `./output/`:

- **`logicnet.v`**: Top-level module instantiating all layers
- **`layer0.v`, `layer1.v`**: Layer modules
- **`layer0_N0.v`, `layer0_N1.v`, ...**: Individual neuron LUT modules
- **Vivado project** (if synthesis enabled): Resource utilization reports

## Understanding the Verilog

### Neuron Module (LUT)

Each neuron is implemented as a ROM-based lookup table:

```verilog
module layer0_N0 (
    input [3:0] M0,      // 2 inputs × 2 bits each = 4 bits
    output [1:0] M1      // 2-bit output
);
    (*rom_style = "distributed"*) reg [1:0] M1r;
    assign M1 = M1r;
    always @ (M0) begin
        case (M0)
            4'b0000: M1r = 2'b00;
            4'b0001: M1r = 2'b01;
            // ... 16 total entries (2^4)
        endcase
    end
endmodule
```

### Layer Module

Instantiates multiple neurons with sparse connectivity:

```verilog
module layer0 (
    input [7:0] M0,      // 4 inputs × 2 bits = 8 bits
    output [5:0] M1      // 3 neurons × 2 bits = 6 bits
);
    // Neuron 0: connects to inputs 0,2
    wire [3:0] layer0_N0_wire = {M0[5:4], M0[1:0]};
    layer0_N0 layer0_N0_inst (.M0(layer0_N0_wire), .M1(M1[1:0]));

    // Neuron 1: connects to inputs 1,3
    wire [3:0] layer0_N1_wire = {M0[7:6], M0[3:2]};
    layer0_N1 layer0_N1_inst (.M0(layer0_N1_wire), .M1(M1[3:2]));

    // ... more neurons
endmodule
```

### Top Module

Connects layers with optional pipeline registers:

```verilog
module logicnet (
    input [7:0] M0,      // Network input
    input clk,
    input rst,
    output [3:0] M2      // Network output (2 outputs × 2 bits)
);
    wire [5:0] M1;       // Hidden layer output
    layer0 layer0_inst (.M0(M0), .M1(M1));
    layer1 layer1_inst (.M0(M1), .M1(M2));
endmodule
```

## FPGA Resource Estimation

### Theoretical LUT Cost Formula

From the FPL'20 paper:

```
LUTCost(X, Y) = (Y / 3) * (2^(X - 4) - (-1)^X)

where:
  X = input fan-in bits (connections × bitwidth)
  Y = output bits
```

For this minimal example:
- **Layer 0**: Each of 3 neurons has X=4 (2 inputs × 2 bits), Y=2
  - Per neuron: (2/3) × (2^0 - 1) = 0.67 × 0 = ~1 LUT (minimum)
  - Layer total: ~3 LUTs
- **Layer 1**: Each of 2 neurons has X=4, Y=2
  - Layer total: ~2 LUTs

**Estimated total**: ~5 LUTs

### Actual Synthesis

Vivado synthesis typically reports:
- ~5-10 LUTs (close to theoretical estimate)
- ~5-10 Flip-Flops (for pipelining if enabled)
- Maximum frequency: 200-400 MHz (depending on FPGA and registers)

## Key Insights

1. **Sparsity reduces LUT cost exponentially**: Each neuron's LUT size is 2^(fan-in × bitwidth). With fan-in=2 and 2-bit inputs, we only need 2^4=16 LUT entries per neuron.

2. **Quantization is essential**: 2-bit quantization means each connection adds only 2 bits to the LUT input, not 32 bits of float32.

3. **No arithmetic operations**: The entire NN executes as pure lookup tables—no multipliers, no adders, just ROM reads and wire connections.

4. **Throughput vs Latency trade-off**:
   - Without registers: 1 cycle latency, lower fmax
   - With registers: N-layer cycle latency, higher fmax

## Next Steps

To explore further:

1. **Modify the architecture** in `model.py`:
   - Try different layer sizes
   - Adjust fan-in (2 → 3 or 4)
   - Change bitwidths (2 → 3 or 4)

2. **Compare synthesis results**: See how LUT cost scales with fan-in and bitwidth

3. **Train a real model**: Use the `jet_substructure` or `cybersecurity` examples to see trained networks

4. **Simulate Verilog**: Install PyVerilator to run functional simulation of generated HDL

## References

- LogicNets paper: https://arxiv.org/abs/2004.03021
- Brevitas quantization: https://github.com/Xilinx/brevitas
- oh-my-xilinx synthesis: https://github.com/Xilinx/oh-my-xilinx
