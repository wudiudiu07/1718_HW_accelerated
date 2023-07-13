`timescale 1 ns /1 ps
module tb(); 
  
  parameter BUS_WIDTH = 32; 
  parameter CLK_PERIOD = 20; 
  parameter N_INPUT = 49; // filter_size x filter_size
  localparam TEST_INPUT = {"./input.txt"};
  localparam GOLD_RES = {"./output.txt"};
  
  logic clk, rst;
  logic [1:0] control; 
  logic [8000:0] line;
  logic [ BUS_WIDTH-1 : 0 ] bus_r; 
  wire [ BUS_WIDTH-1 : 0 ] bus; 

  //TEST INPUT
  //logic [BUS_WIDTH-1 : 0] conv_w [N_INPUT-1 :0];
  logic signed [BUS_WIDTH-1:0] conv_w [N_INPUT-1 :0];
  logic signed [BUS_WIDTH-1 : 0] img [N_INPUT-1 :0];
  logic signed [BUS_WIDTH-1 : 0] sum;
  integer res_sum;
  logic ival, oval, o_wready, o_iready;
  
  //CODE START
  integer i,j, cnt, nmatch_cnt;
  integer temp, in_file, res_file;
  assign bus = &control ? {BUS_WIDTH{1'bZ}} : bus_r; 


  conv_top #(.BUS_WIDTH(BUS_WIDTH),.N_INPUT(N_INPUT)) DUT ( 
    .clk(clk), 
    .rst(rst),
    .i_ctrl(control),
    .o_val(oval),
    .o_wreq(o_wready),
    .o_ireq(o_iready),
    .iobus(bus) 
  ); 
  //mac_hw #(BUS_WIDTH, N_INPUT) mac1 (.clk(clk), .rst(rst), .i_val(ival), .i_Weight(conv_w), .i_Img(img), .o_sum(sum), .o_val(oval));
 initial clk = 1'b1; 
 always #(CLK_PERIOD/2) clk = ~clk; 
 
  initial begin
	 
     in_file = $fopen(TEST_INPUT, "r");
     res_file = $fopen(GOLD_RES, "r");
     // read weight once
     //
      temp = $fgets(line, in_file);
      temp = $sscanf(line, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d ", 
			conv_w[0], conv_w[1], conv_w[2],conv_w[3], conv_w[4], conv_w[5],conv_w[6], conv_w[7], conv_w[8],conv_w[9], conv_w[10], conv_w[11],conv_w[12], conv_w[13], conv_w[14],conv_w[15], conv_w[16], conv_w[17],
		conv_w[18], conv_w[19], conv_w[20],conv_w[21], conv_w[22], conv_w[23],conv_w[24], conv_w[25], conv_w[26],conv_w[27], conv_w[28], conv_w[29],conv_w[30], conv_w[31], conv_w[32],conv_w[33], conv_w[34], conv_w[35],
	conv_w[36], conv_w[37], conv_w[38],conv_w[39], conv_w[40], conv_w[41],conv_w[42], conv_w[43], conv_w[44],conv_w[45], conv_w[46], conv_w[47],conv_w[48]);
     
     cnt = 0;
     nmatch_cnt = 0;
     rst = 1'b1;
     #(3*CLK_PERIOD);
     rst = 1'b0;

     // load weight once
     for (i=0;i<N_INPUT;i++)begin
     	#(CLK_PERIOD);
	while(!o_wready) #(CLK_PERIOD);
	control = 2'd1;
	bus_r = conv_w[i];	
     end
     
     #(CLK_PERIOD);
     control = 2'd0;
     while (!$feof(in_file)) begin
	// read img from file
      temp = $fgets(line, in_file);
      temp = $sscanf(line, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d ", 
			img[0], img[1], img[2],img[3], img[4], img[5],img[6], img[7], img[8],img[9], img[10], img[11],img[12], img[13], img[14],img[15], img[16], img[17],
		img[18], img[19], img[20],img[21], img[22], img[23],img[24], img[25], img[26],img[27], img[28], img[29],img[30], img[31], img[32],img[33], img[34], img[35],
	img[36], img[37], img[38],img[39], img[40], img[41],img[42], img[43], img[44],img[45], img[46], img[47],img[48]);
	   
//          
//	  #(3*CLK_PERIOD/2);
//	  while(!oval) begin
//		  ival = 1'b0;
//	  	#(3*CLK_PERIOD/2);
//	  end
//	
//         temp = $fscanf(res_file, "%d ", res_sum);	
//       	 if (res_sum == sum)	 
//	 	$display("result %d: %d %d, MATCH!", cnt++, sum, res_sum); 
//	else begin
//	 	$display("result %d: %d %d, DONT MATCH!", cnt++, sum, res_sum); 
//		nmatch_cnt ++;
//	end
//
	  for (i=0;i<N_INPUT;i++)begin
	     #(CLK_PERIOD);
	     while(!o_iready) #(CLK_PERIOD);
	     control = 2'd2;    // load second operand 
	     bus_r = img[i];
	   end 
	  #(CLK_PERIOD)
	 control = 2'd0;
	 while(!oval)#(CLK_PERIOD); 
	  control =2'd3;
	 #(CLK_PERIOD)  // test result 
         temp = $fscanf(res_file, "%d ", res_sum);	
      	 if (res_sum == bus)	 
	 	$display("result %d: %d %d, MATCH!", cnt++, bus, res_sum); 
	else begin
	 	$display("result %d: %d %d, DONT MATCH!", cnt++, bus, res_sum); 
		nmatch_cnt ++;
	end
	 //$display("result %d: %d", cnt++, bus); 
	 control = 2'b00;

     end

      if (nmatch_cnt == 0)$display("ALL MATCHED, TEST PASS!");
      else $display("%d SUM DONT MATCH, TEST FAILED!", nmatch_cnt);
      $fclose(in_file);
  end 
endmodule 


