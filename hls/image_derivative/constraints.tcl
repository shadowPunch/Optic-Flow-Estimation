set_directive_pipeline "image_derivative_kernel/col_loop" -II 1
set_directive_array_partition "image_derivative_kernel" line_buffer -type complete -dim 1
set_directive_array_partition "image_derivative_kernel" window -type complete
set_directive_resource "image_derivative_kernel" line_buffer -core RAM_S2P_BRAM

set_directive_interface "image_derivative_kernel" input_stream -mode axis
set_directive_interface "image_derivative_kernel" dx_stream -mode axis  
set_directive_interface "image_derivative_kernel" dy_stream -mode axis
set_directive_interface "image_derivative_kernel" width -mode s_axilite
set_directive_interface "image_derivative_kernel" height -mode s_axilite
set_directive_interface "image_derivative_kernel" return -mode s_axilite

config_op mul -impl fabric
config_op add -impl fabric