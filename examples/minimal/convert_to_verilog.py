#!/usr/bin/env python3
#  Minimal example: Convert a randomly initialized NN to Verilog and get FPGA resource estimates

import os
import torch
from argparse import ArgumentParser

from model import MinimalNeqModel, MinimalLutModel
from logicnets.nn import generate_truth_tables, lut_inference, module_list_to_verilog_module
from logicnets.synthesis import synthesize_and_get_resource_counts
from logicnets.util import get_lut_cost


def main():
    parser = ArgumentParser(description="Convert minimal NN to Verilog and estimate FPGA resources")
    parser.add_argument('--output-dir', type=str, default='./output',
                       help="Directory to store generated Verilog (default: %(default)s)")
    parser.add_argument('--clock-period', type=float, default=10.0,
                       help="Target clock period in ns for synthesis (default: %(default)s)")
    parser.add_argument('--fpga-part', type=str, default="xc7a35tcpg236-1",
                       help="FPGA part number for synthesis (default: %(default)s)")
    parser.add_argument('--skip-synthesis', action='store_true', default=False,
                       help="Skip Vivado synthesis (only generate Verilog)")
    parser.add_argument('--add-registers', action='store_true', default=False,
                       help="Add pipeline registers between layers")
    args = parser.parse_args()

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("=" * 70)
    print("MINIMAL LOGICNETS EXAMPLE: NN to Verilog Translation")
    print("=" * 70)

    # Step 1: Instantiate the model
    print("\n[Step 1] Creating minimal neural network...")
    print("  Architecture: 4 inputs -> 3 hidden neurons -> 2 outputs")
    print("  Quantization: 2-bit throughout")
    print("  Sparsity: Fan-in of 2 (each neuron connects to 2 inputs)")

    neq_model = MinimalNeqModel()
    neq_model.eval()

    # Print network architecture
    print("\n  Network structure:")
    for i, layer in enumerate(neq_model.module_list):
        print(f"    Layer {i}: {layer.in_features} -> {layer.out_features}")
        print(f"      Input bitwidth: {layer.input_quant.fused_activation_quant_proxy.tensor_quant.int_quant.narrow_range}")
        print(f"      Output bitwidth: {layer.output_quant.fused_activation_quant_proxy.tensor_quant.int_quant.narrow_range}")

    # Step 2: Test the PyTorch model with random input
    print("\n[Step 2] Testing PyTorch model with random input...")
    test_input = torch.randn(1, 4)
    pytorch_output = neq_model(test_input)
    print(f"  Input:  {test_input.detach().numpy()}")
    print(f"  Output: {pytorch_output.detach().numpy()}")

    # Step 3: Estimate LUT cost
    print("\n[Step 3] Estimating FPGA LUT cost...")
    lut_cost = get_lut_cost(neq_model)
    print(f"  Estimated LUT cost: {lut_cost:.0f} LUTs")
    print("  (This is a theoretical estimate based on the formula from the FPL'20 paper)")

    # Step 4: Create LUT-based model and generate truth tables
    print("\n[Step 4] Converting NEQ model to LUT-based representation...")
    lut_model = MinimalLutModel()
    lut_model.load_state_dict(neq_model.state_dict())
    lut_model.eval()

    print("  Generating truth tables for each neuron...")
    generate_truth_tables(lut_model, verbose=True)
    print("  Truth tables generated!")

    # Step 5: Test LUT-based inference
    print("\n[Step 5] Testing LUT-based inference (should match PyTorch)...")
    lut_inference(lut_model)
    lut_output = lut_model(test_input)
    print(f"  Input:      {test_input.detach().numpy()}")
    print(f"  LUT Output: {lut_output.detach().numpy()}")
    print(f"  PyTorch Output: {pytorch_output.detach().numpy()}")

    # Check if outputs match
    if torch.allclose(lut_output, pytorch_output, atol=1e-5):
        print("  ✓ LUT inference matches PyTorch!")
    else:
        print("  ✗ Warning: LUT inference differs from PyTorch")
        print(f"    Difference: {torch.abs(lut_output - pytorch_output).max().item()}")

    # Step 6: Generate Verilog
    print("\n[Step 6] Generating Verilog HDL...")
    module_list_to_verilog_module(
        lut_model.module_list,
        "logicnet",
        args.output_dir,
        generate_bench=False,
        add_registers=args.add_registers
    )
    print(f"  ✓ Verilog generated in: {args.output_dir}/")
    print(f"    Top module: {args.output_dir}/logicnet.v")

    # List generated files
    verilog_files = [f for f in os.listdir(args.output_dir) if f.endswith('.v')]
    print(f"  Generated {len(verilog_files)} Verilog files:")
    for vf in sorted(verilog_files):
        file_size = os.path.getsize(os.path.join(args.output_dir, vf))
        print(f"    - {vf} ({file_size} bytes)")

    # Step 7: Synthesize with Vivado (optional)
    if not args.skip_synthesis:
        print("\n[Step 7] Running Vivado synthesis...")
        print(f"  Target FPGA: {args.fpga_part}")
        print(f"  Clock period: {args.clock_period} ns")
        print("  (This may take 1-2 minutes...)")

        try:
            results = synthesize_and_get_resource_counts(
                args.output_dir,
                "logicnet",
                fpga_part=args.fpga_part,
                clk_period_ns=args.clock_period,
                post_synthesis=0
            )

            print("\n  ✓ Synthesis complete!")
            print("\n  FPGA Resource Utilization:")
            print("  " + "-" * 40)
            for resource, value in results.items():
                print(f"    {resource:20s}: {value}")
            print("  " + "-" * 40)

            # Compare with theoretical estimate
            if 'LUTs' in results:
                actual_luts = int(results['LUTs'])
                estimated_luts = int(lut_cost)
                print(f"\n  Theoretical estimate: {estimated_luts} LUTs")
                print(f"  Actual synthesis:     {actual_luts} LUTs")
                if actual_luts > 0:
                    ratio = estimated_luts / actual_luts
                    print(f"  Ratio: {ratio:.2f}x")

        except Exception as e:
            print(f"\n  ✗ Synthesis failed: {e}")
            print("  Note: Vivado and oh-my-xilinx must be installed and in PATH")
    else:
        print("\n[Step 7] Skipping synthesis (--skip-synthesis specified)")

    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  - Verilog files generated in: {args.output_dir}/")
    print(f"  - Estimated LUT cost: {lut_cost:.0f} LUTs")
    if not args.skip_synthesis:
        print(f"  - Synthesis results available in: {args.output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
