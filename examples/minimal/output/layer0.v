module layer0 (input [7:0] M0, output [5:0] M1);

wire [3:0] layer0_N0_wire = {M0[1], M0[0], M0[5], M0[4]};
layer0_N0 layer0_N0_inst (.M0(layer0_N0_wire), .M1(M1[1:0]));

wire [3:0] layer0_N1_wire = {M0[1], M0[0], M0[5], M0[4]};
layer0_N1 layer0_N1_inst (.M0(layer0_N1_wire), .M1(M1[3:2]));

wire [3:0] layer0_N2_wire = {M0[1], M0[0], M0[5], M0[4]};
layer0_N2 layer0_N2_inst (.M0(layer0_N2_wire), .M1(M1[5:4]));

endmodule