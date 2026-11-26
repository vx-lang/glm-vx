# z64b (#63, TTFT): the maintainer asked for only the 30K prompts of z64's passkey check (2026-11-21 06:41 UTC), on
# z64's branch server (repo-z63, still up), then the server is stopped.
set -u
cd ~/glm-h200
. ./env.sh
SG=$HOME/glm-h200/sglang-venv/bin/python M=$HOME/glm-h200/GLM-5.3 out=$HOME/glm-h200/z64-out
cp z64b.sh passkey30.py $out/
timeout 3600 $SG passkey30.py http://127.0.0.1:8000 $M/tokenizer.json $out/ttft-passkey30.json > $out/ttft-passkey30.log 2>&1
echo "ttft-passkey30: rc=$? $(date -u +%T): $(tail -n 1 $out/ttft-passkey30.log | cut -c1-300)"
pkill -f "[g]lm_vx.server --backend"; sleep 20; pkill -9 -f "[g]pu.tp_serve --checkpoint"; pkill -9 -f "[g]lm_vx.server --backend"
echo z64b-done
