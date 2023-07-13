vlib work
vlog -novopt tb_conv_hw.sv conv_hw.sv mac_hw.sv
vsim  -displaymsgmode both work.tb
add wave  sim:/tb/DUT/*
add wave  sim:/tb/DUT/mac1/*
add wave  -decimal sim:/tb/cnt
run -all
