module layer1 (input [5:0] M0, output [3:0] M1);

wire [3:0] layer1_N0_wire = {M0[1], M0[0], M0[3], M0[2]};
layer1_N0 layer1_N0_inst (.M0(layer1_N0_wire), .M1(M1[1:0]));

wire [3:0] layer1_N1_wire = {M0[1], M0[0], M0[5], M0[4]};
layer1_N1 layer1_N1_inst (.M0(layer1_N1_wire), .M1(M1[3:2]));

endmodule