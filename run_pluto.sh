#!/bin/bash
cd /home/kevin/claude-code/mother-embedding
julia --project=. -e 'using Pluto; Pluto.run(host="0.0.0.0", port=1235, launch_browser=false, notebook="View_GaborKernels.jl")'
