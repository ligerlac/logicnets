module logicnet (input [7:0] M0, input clk, input rst, output[3:0] M2);
wire [7:0] M0w;
assign M0w = M0;
wire [5:0] M1;
layer0 layer0_inst (.M0(M0w), .M1(M1));
wire [5:0] M1w;
assign M1w = M1;
layer1 layer1_inst (.M0(M1w), .M1(M2));

endmodule
