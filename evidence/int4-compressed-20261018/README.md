Three INT4 Linears from `lowbitcoffee/GLM-5.2-W4A16` (glm_moe_dsa, compressed-tensors pack-quantized),
fetched by range request (not committed; re-fetch to reproduce, the hash is in fixture.sha256):

    python -m validation.fetch_tensors https://huggingface.co/lowbitcoffee/GLM-5.2-W4A16/resolve/main/model-00002-of-00008.safetensors DIR \
        model.layers.16.self_attn.kv_a_proj_with_mqa. model.layers.16.mlp.experts.0.down_proj. model.layers.16.self_attn.kv_b_proj.
    curl -L https://huggingface.co/lowbitcoffee/GLM-5.2-W4A16/raw/main/config.json -o DIR/config.json

`oracle.json`: `python -m validation.compressed_int4_oracle DIR` under torch 2.x CPU + compressed-tensors 0.19.0.
Codes equal and bf16 weights bitwise equal to the library for all three. With
`GLM_VX_INT4_CHECKPOINT=DIR`, `gpu/test_quant_simt.py::test_int4_real_checkpoint_linears` (CPU SIMT,
passed locally) and `gpu/test_quant_hardware.py::test_int4_real_checkpoint_linears_device` (A100, not
run yet: no GPU) run the kernel on these weights.
