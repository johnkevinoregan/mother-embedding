#!/usr/bin/env bash
# Runnable commands for SimpleStrokeTests.  Usage:  ./run.sh <name>
#
#   ./run.sh preview     figures + dataset audit          ~4 min
#   ./run.sh fast        every arm except the CNN         ~20 min
#   ./run.sh full        all five arms, GPU, big CNN      ~40 min
#   ./run.sh figures     redraw figures from saved results
#   ./run.sh repl        interactive Julia in this project
#
# Add `bg` as a second argument to run it in the background and log to a file:
#   ./run.sh full bg
#
# Each command is echoed before it runs, so you can copy it out of the log if you
# want to vary it by hand.

set -euo pipefail
cd "$(dirname "$0")"

BG="${2:-}"

run() {
    echo; echo "\$ $1"; echo
    if [[ "$BG" == "bg" ]]; then
        local log="bg_${1:0:0}${CMD}.log"
        nohup bash -c "$1" > "$log" 2>&1 &
        echo "running in the background as pid $!, logging to $log"
        echo "watch it with:   tail -f $(pwd)/$log"
        echo "stop it with:    pkill -f Phase9_Readouts"
    else
        eval "$1"
    fi
}

CMD="${1:-help}"
case "$CMD" in

preview)
    run "julia --project=.. Preview_Contours.jl" ;;

fast)
    # Linear and MLP arms: the whole measurement plus the shuffle control.
    run "P9_NTRAIN=12000 P9_NTEST=3000 P9_KS=500,2000,6000,12000 \
         P9_EPOCHS=60 P9_ARMS=1,2,4,5 P9_OUT=results_nocnn \
         julia --project=.. -t 16 Phase9_Readouts.jl 2>&1 | tee nocnn.log" ;;

full)
    # All five arms. Uses the GPU automatically if there is one; set P9_GPU=0 to force CPU.
    run "P9_NTRAIN=12000 P9_NTEST=3000 P9_KS=500,2000,6000,12000 \
         P9_EPOCHS=60 P9_CEPOCHS=60 P9_CNN=big P9_CURVE_ARMS=1,3,4 P9_OUT=results_gpu \
         julia --project=.. -t 12 Phase9_Readouts.jl 2>&1 | tee gpu.log" ;;

figures)
    run "julia --project=.. Plot_Phase9.jl" ;;

repl)
    echo
    echo "Starting Julia in this project. Once at the julia> prompt:"
    echo '    include("Preview_Contours.jl")'
    echo "Ctrl-D to leave."
    echo
    exec julia --project=.. -t 8 ;;

*)
    sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
