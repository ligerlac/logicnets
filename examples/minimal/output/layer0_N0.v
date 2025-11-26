module layer0_N0 ( input [3:0] M0, output [1:0] M1 );

	(*rom_style = "distributed" *) reg [1:0] M1r;
	assign M1 = M1r;
	always @ (M0) begin
		case (M0)
			4'b0000: M1r = 2'b00;
			4'b0100: M1r = 2'b00;
			4'b1000: M1r = 2'b00;
			4'b1100: M1r = 2'b00;
			4'b0001: M1r = 2'b00;
			4'b0101: M1r = 2'b00;
			4'b1001: M1r = 2'b00;
			4'b1101: M1r = 2'b00;
			4'b0010: M1r = 2'b00;
			4'b0110: M1r = 2'b00;
			4'b1010: M1r = 2'b00;
			4'b1110: M1r = 2'b00;
			4'b0011: M1r = 2'b00;
			4'b0111: M1r = 2'b00;
			4'b1011: M1r = 2'b00;
			4'b1111: M1r = 2'b01;

		endcase
	end
endmodule
