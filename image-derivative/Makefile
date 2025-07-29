VITIS_HLS = vitis_hls
PROJECT = image_derivative_prj

.PHONY: all csim csynth cosim export clean

all: csynth

csim:
	$(VITIS_HLS) -f script.tcl -tclargs csim

csynth:
	$(VITIS_HLS) -f script.tcl -tclargs csynth

cosim:
	$(VITIS_HLS) -f script.tcl -tclargs cosim

export:
	$(VITIS_HLS) -f script.tcl -tclargs export

clean:
	rm -rf $(PROJECT)
	rm -rf .Xil
	rm -f *.log
	rm -f vivado*.jou