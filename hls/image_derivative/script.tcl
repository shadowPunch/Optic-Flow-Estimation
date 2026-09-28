open_project image_derivative_prj
set_top image_derivative_kernel
add_files src/image_derivative.cpp
add_files -tb tb/testbench.cpp

open_solution "solution1" -flow_target vivado
set_part {xck26-sfvc784-2LV-c}

config_interface -m_axi_latency=64 -m_axi_max_widen_bitwidth=512
config_compile -pipeline_loops=64
config_schedule -enable_dsp_full_reg=false

csynth_design
cosim_design -trace_level all
export_design -format ip_catalog -vendor "xilinx.com" -library "hls" -version "1.0"

exit