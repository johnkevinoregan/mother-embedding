#!/bin/bash
# Runs the rest of Phase 13 unattended: few-shot evaluation as soon as its features land,
# then all figures once the last from-scratch run finishes.
cd "$(dirname "$0")" || exit 1
set -x
until [ -f fewshot/scratch_test.f32 ]; do sleep 30; done
sleep 10
julia --project=.. -t 14 FewShot_Eval.jl > fewshot_eval.log 2>&1
julia --project=.. Plot_FewShot.jl      > fewshot_plot.log 2>&1
until [ -f scratch_runs/base224_set1.tsv ]; do sleep 60; done
sleep 10
julia --project=.. Plot_Curriculum.jl   > plots.log 2>&1
echo "ALL DONE"
