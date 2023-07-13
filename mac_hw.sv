`timescale 1 ns /1 ps
module mac_hw_pl #(
  parameter WIDTH = 32,
  parameter N_INPUT = 49
)(
  input logic clk,
  input logic rst,
  input logic i_val,
  input logic signed [WIDTH-1:0] i_Weight [N_INPUT-1:0],
  input logic signed [WIDTH-1:0] i_Img [N_INPUT-1:0],
  output logic [WIDTH-1:0] o_sum,
  output logic o_val
  
);
  logic signed [WIDTH-1:0] mul [2**$clog2(N_INPUT)-1:0];
  logic signed [WIDTH-1:0] sum, sum_d;
  logic signed [WIDTH-1:0] sum_v [$clog2(N_INPUT)-1:0][2**($clog2(N_INPUT)-1)-1:0];
  logic [$clog2(N_INPUT):0]val_d;
  integer k;

  assign o_sum = sum_d;


  genvar i, j;
  generate
  for (i=0;i<2**$clog2(N_INPUT);i++) begin : multi
        always_ff@(posedge clk) begin
          if (rst)
            mul[i] <= 0;
	  else if (i<N_INPUT)
 	    mul[i] <= $signed(i_Weight[i]) * $signed(i_Img[i]);
          else
            mul[i] <= 0;
        end
  end
  endgenerate

  generate
    if ($clog2(N_INPUT)>0) begin
      for(i=0;i<$clog2(N_INPUT);i++) begin : acc
        for (j=0;j<2**($clog2(N_INPUT)-i-1);j++) begin : acc_sub
          always_ff@(posedge clk) begin
            if (rst) begin
              sum_v[i][j] <= 0;
            end else begin
              if (i==0)begin 
                  sum_v[i][j] <= $signed(mul[j*2]) + $signed(mul[j*2+1]);
              end else 
                  sum_v[i][j] <= $signed(sum_v[i-1][j*2]) + $signed(sum_v[i-1][j*2+1]); 
            end
          end
        end
      end
      assign sum_d = sum_v[$clog2(N_INPUT)-1][0];
    end else
      assign sum_d = mul[0];
  endgenerate
  
  assign o_val = val_d[$clog2(N_INPUT)];

  always_ff@(posedge clk) begin
    if (rst) begin
      val_d <= 0;
    end else begin
      if ($clog2(N_INPUT)>0) begin
        for (k=1;k<=$clog2(N_INPUT);k++)
          val_d[k] <= val_d[k-1];
      end
      //val_d[0] <= val_m;
      val_d[0] <= i_val;
    end

  end

endmodule




module mac_hw #(
  parameter WIDTH = 32,
  parameter N_INPUT = 49
)(
  input logic clk,
  input logic rst,
  input logic i_val,
  input logic signed [WIDTH-1:0] i_Weight [N_INPUT-1:0],
  input logic signed [WIDTH-1:0] i_Img [N_INPUT-1:0],
  output logic [WIDTH-1:0] o_sum,
  output logic o_val
  
);
  logic signed [WIDTH-1:0] mul [2**$clog2(N_INPUT)-1:0];
  logic signed [WIDTH-1:0] sum, sum_d;
  logic val_d;
  integer j;

  assign o_sum = sum_d;
  always_ff@(posedge clk)begin
      if (rst) begin
        o_val <= 1'b0;
        val_d <= 1'b0;
      end else begin
        val_d <= i_val;
        o_val <= val_d;
	if (val_d) begin
          sum_d <= sum;
          //o_mul_v <= mul;
	end
      end
  end
  
  always_comb begin
      j=0;
      if (N_INPUT == 1)begin
        sum = mul[0];
      end else begin
        sum = 0;
        // can be optimized
	if (val_d) begin
        	for (j=0;j<N_INPUT;j++) begin
          		sum = $signed(sum) + $signed(mul[j]);
        	end
	end
      end
  end


  genvar i;
  generate
  for (i=0;i<N_INPUT;i++) begin : multi
        always_ff@(posedge clk) begin
          if (rst)
            mul[i] <= 0;
	  else if (i_val)
 	    mul[i] <= $signed(i_Weight[i]) * $signed(i_Img[i]);
        end
  end
  endgenerate


endmodule

