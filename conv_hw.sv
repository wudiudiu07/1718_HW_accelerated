
//module mac_hw #(
//  parameter WIDTH = 32,
//  parameter N_INPUT = 1
//)(
//  input logic clk,
//  input logic rst,
//  input logic i_val,
//  input logic signed [WIDTH-1:0] i_Weight [N_INPUT-1:0],
//  input logic signed [WIDTH-1:0] i_Img [N_INPUT-1:0],
//  output logic [WIDTH-1:0] o_sum,
//  output logic o_val
//  
//);
//  logic signed [WIDTH-1:0] mul [2**$clog2(N_INPUT)-1:0];
//  logic signed [WIDTH-1:0] sum, sum_d;
//  logic signed [WIDTH-1:0] sum_v [$clog2(N_INPUT)-1:0][2**($clog2(N_INPUT)-1)-1:0];
//  logic [$clog2(N_INPUT):0]val_d;
//  logic val_m;
//  integer k;
//
//  assign o_sum = sum_d;
////  always_ff@(posedge clk)begin
////      if (rst) begin
////        o_val <= 1'b0;
////        val_d <= 1'b0;
////      end else begin
////        val_d <= i_val;
////        o_val <= val_d;
////	if (val_d) begin
////          sum_d <= sum;
////          //o_mul_v <= mul;
////	end
////      end
////  end
////  always_comb begin
////      if (N_INPUT == 1)begin
////        sum = mul[0];
////      end else begin
////        sum = 0;
////        // can be optimized
////	if (val_d) begin
////        	for (j=0;j<N_INPUT;j++) begin
////          		sum = $signed(sum) + $signed(mul[j]);
////        	end
////	end
////      end
////  end
////
////
//
//  genvar i, j;
//  generate
//  for (i=0;i<2**$clog2(N_INPUT);i++) begin : multi
//        always_ff@(posedge clk) begin
//          if (rst)
//            mul[i] <= 0;
//	  else if (i<N_INPUT)
// 	    mul[i] <= $signed(i_Weight[i]) * $signed(i_Img[i]);
//          else
//            mul[i] <= 0;
//        end
//  end
//  endgenerate
//
//  generate
//    if ($clog2(N_INPUT)>0) begin
//      for(i=0;i<$clog2(N_INPUT);i++) begin : acc
//        for (j=0;j<2**($clog2(N_INPUT)-i-1);j++) begin : acc_sub
//          always_ff@(posedge clk) begin
//            if (rst) begin
//              sum_v[i][j] <= 0;
//            end else begin
//              if (i==0)begin 
//                  sum_v[i][j] <= $signed(mul[j*2]) + $signed(mul[j*2+1]);
//              end else 
//                  sum_v[i][j] <= $signed(sum_v[i-1][j*2]) + $signed(sum_v[i-1][j*2+1]); 
//            end
//          end
//        end
//      end
//      assign sum_d = sum_v[$clog2(N_INPUT)-1][0];
//    end else
//      assign sum_d = mul[0];
//  endgenerate
//  
//  assign o_val = val_d[$clog2(N_INPUT)];
//
//  always_ff@(posedge clk) begin
//    if (rst) begin
//      val_d <= 0;
//    end else begin
//      if ($clog2(N_INPUT)>0) begin
//        for (k=1;k<=$clog2(N_INPUT);k++)
//          val_d[k] <= val_d[k-1];
//      end
//      //val_d[0] <= val_m;
//      val_d[0] <= i_val;
//    end
//
//  end
//
//endmodule

module conv_top #(
  parameter BUS_WIDTH = 32,
  parameter N_INPUT = 1
)(
  input logic clk,
  input logic rst,
  input logic [1:0] i_ctrl,
  output logic o_val,
  output logic o_wreq,
  output logic o_ireq,
  inout logic [BUS_WIDTH-1:0] iobus

);
  //localparam N_INPUT = 1;
  localparam D_WIDTH = BUS_WIDTH;
  enum int unsigned {s_IDLE, s_LOAD_W, s_LOAD_I, s_CAL} state;
  
  logic [5:0]  w_cnt, i_cnt;
  logic signed [D_WIDTH-1:0] conv_w [N_INPUT-1:0]; 
  logic signed [D_WIDTH-1:0] img [N_INPUT-1:0]; 
  logic [1:0] readyi;
  logic [D_WIDTH-1:0] sum, sumd;
  logic [BUS_WIDTH-1:0] obus;
  logic ival, oval, wreq, ireq, val_d;

  integer i;
  //mac_hw #(D_WIDTH, N_INPUT) mac1 (.clk(clk), .rst(rst), .i_val(ival), .i_Weight(conv_w), .i_Img(img), .o_sum(sum), .o_val(oval));
  mac_hw_pl #(D_WIDTH, N_INPUT) mac1 (.clk(clk), .rst(rst), .i_val(ival), .i_Weight(conv_w), .i_Img(img), .o_sum(sum), .o_val(oval));
  assign iobus = (&i_ctrl) ? sumd : {BUS_WIDTH{1'bZ}};
  assign o_wreq = wreq;
  assign o_ireq = ireq;
  assign o_val = val_d;
  // bus to weight, img buffer 
  always_ff@(posedge clk) begin

      if (i_ctrl == 2'b01 && wreq) begin
        conv_w[0] <= iobus;
        for(i=1;i<N_INPUT; i++)
          conv_w[i]<=conv_w[i-1];
      end else if (i_ctrl == 2'b10 && ireq) begin
        img[0] <= iobus;
        for(i=1;i<N_INPUT; i++)
          img[i]<=img[i-1];
      end
  
  end

  always_ff@(posedge clk) begin
    if (rst) begin
      val_d <= 1'b0;
      ival <= 1'b0;
      sumd <= 0;
      w_cnt <= 0;
      i_cnt <= 0;
      state <= s_IDLE;
    end else begin
      ival <= 1'b0;
      wreq <= 1'b0;
      ireq <= 1'b0;
      case(state)
      
        s_IDLE  : begin
          val_d <= 1'b0;
          wreq <= 1'b1;
          if (i_ctrl == 2'd1)begin
            state <= s_LOAD_W;
            //w_cnt <= w_cnt + 1;
          end
        end
        s_LOAD_W: begin
          val_d <= 1'b0;
          wreq <= 1'b1;
          if (i_ctrl == 2'd1)w_cnt <= w_cnt + 1;
          
          if (w_cnt == N_INPUT-2) begin
            state <= s_LOAD_I;
            ireq <= 1'b1;
            wreq <= 1'b0;
            w_cnt <= 0;
          end
        end
        s_LOAD_I: begin
          val_d <= 1'b0;
          ireq <= 1'b1;
          if (i_ctrl == 2'd2)i_cnt <= i_cnt + 1;
          
          if (i_cnt == N_INPUT-1)begin
            ival <= 1'b1;
            ireq <= 1'b0;
            i_cnt <= 0;
            state <= s_CAL;
          end
        end

        s_CAL: begin
          if(oval)begin
            sumd <= sum;
            val_d <= 1'b1;
          end
          if(i_ctrl == 2'd3)
            state <= s_LOAD_I;
        end
      endcase

    end
  end	  

endmodule



