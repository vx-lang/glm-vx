# H200 session z3: (A) Vx PR #3609, the arrivals' count ordered for the device; (B) glm-vx against SGLang
# 0.5.22 at batch 1, the same contexts, the same day.
# Copy repo-z3.tar.gz, spawn-cb0f.tar.gz, spawn-3609.tar.gz and z3.sh to ~/glm-h200 first. Needs from z2:
# repo-z2 (its gpu/build, Vx 6a1d), env-z2.sh, rb5/ (router_bf16 unrolled, 6a1d), rb5nf/ (the same, .sys fences deleted).
set -u
cd ~/glm-h200
for v in cb0f 3609; do rm -rf s$v && mkdir s$v && tar xzf spawn-$v.tar.gz -C s$v; done
rm -rf repo-z3 && tar xzf repo-z3.tar.gz && cp -a repo-z2/gpu/build repo-z3/gpu/
sed -e 's/repo-z2/repo-z3/' -e 's/REPO_REV=[^ ]*/REPO_REV=pod-z3/' env-z2.sh > env-z3.sh
. ./env-z3.sh
cd repo-z3
out=evidence/h200-$(date -u +%Y%m%d)-z3
mkdir -p $out
cp ../z3.sh $out/
# ---- A. Vx #3609: one source (router_bf16 unrolled, the top-k kernels) from 6a1d, main cb0f0428e, and #3609
# (88098c90d = cb0f0428e + the fix). cb0f -> 3609 is the fix alone.
for v in rb5 rb5nf scb0f s3609; do echo "== $v"; timeout 300 $PYTHON ../repo-z2/router_split.py ../$v | grep router_bf16; done > $out/router-3609.log 2>&1
cat $out/router-3609.log
for v in cb0f 3609; do
  timeout 1200 $PYTHON -m validation.topk_hist_ab --spawn ../s$v --output $out/topk-$v.md > $out/topk-$v.log 2>&1; echo "topk $v: $?"; cat $out/topk-$v.log
done
# ---- B. Against SGLang. Ours: production options (graph, cvt, x4, tc MLA, ilp_hist top-k, fold_shared; the router as
# today: bf16_matvec4 + hand router), nsys kernel durations of a replayed step. A dense layer (d1), the same with the
# lm_head and the logits' download (d1-logits), a dense layer and a MoE one (m2, 16 routed experts: a step reads 8 + the
# shared one, as GLM-5.3's 256). So a dense layer = d1, a MoE layer = m2 - d1, the head = d1-logits - d1.
cp ../abdiff.py $out/
prof() { name=$1; shift
  timeout 1500 nsys profile -f true -o $out/$name --capture-range=cudaProfilerApi --capture-range-end=repeat:2 --cuda-graph-trace=node $PYTHON -m validation.step_profile "$@" > $out/$name.log 2>&1
  nsys stats -q --report cuda_gpu_trace --format csv -o $out/$name.2 $out/$name.2.nsys-rep >> $out/$name.log 2>&1; }
for ctx in 1000 3001 32768; do
  prof d1-$ctx --context $ctx --long-topk ilp_hist
  prof d1-$ctx-logits --context $ctx --long-topk ilp_hist --logits
  prof m2-$ctx --context $ctx --long-topk ilp_hist --layers 2 --moe --fold-shared
  for n in d1-$ctx d1-$ctx-logits m2-$ctx; do echo "$n: $($PYTHON $out/abdiff.py $out/$n.2_cuda_gpu_trace.csv $out/$n.2_cuda_gpu_trace.csv 2>&1 | head -n 1)"; done
done
# SGLang: batch-1 decode on the 9-layer FP8 proxy (3 dense + 6 MoE, 256 experts), nsys of its last decode step.
SG_VENV=${SGLANG_VENV:-$(ls -d ~/glm-h200/sglang*venv* 2>/dev/null | head -1)}
SG_PROXY=${PROXY:-$(ls -d ~/glm-h200/proxy-fp8* 2>/dev/null | head -1)}
SG_CACHE=${SGLANG_CACHE_DIR:-$(ls -d ~/glm-h200/sglang-cache* 2>/dev/null | head -1)}
echo "SGLang: venv=$SG_VENV proxy=$SG_PROXY cache=$SG_CACHE"
SGLANG_VENV=$SG_VENV PROXY=$SG_PROXY SGLANG_CACHE_DIR=$SG_CACHE OUT=$PWD/$out/sglang VARIANTS=fp8 PROFILE_BATCHES=1 \
  PROFILE_INPUTS="1000 3001 32768" PROFILER=nsys timeout 3600 bash scripts/pod_sglang.sh profile > $out/sglang.log 2>&1; echo "sglang: $?"
tail -n 5 $out/sglang.log
ls $out/sglang/profile/ 2>/dev/null
echo z3-done
