FROM nvcr.io/nvidia/pytorch:26.07-py3

ENV PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PATH=/opt/ernie/bin:$PATH

# vLLM 0.32 + the b12x SM120/SM121 MXFP4 kernel library + Baidu's ERNIE plugin.
# b12x is the only correct MXFP4 MoE path on GB10; vLLM's auto-selection picks
# Marlin on capability family 120, which silently miscomputes FP4 MoE there.
RUN python3 -m venv /opt/ernie \
 && /opt/ernie/bin/pip install --upgrade pip \
 && /opt/ernie/bin/pip install "vllm[b12x]==0.32.0" \
 && /opt/ernie/bin/pip install "git+https://github.com/baidu/vLLM-ERNIE.git@ba322a5634f91d3e6f96ae711df72ddaf3c38b6e"

# SM121 sink support: route ERNIE's DSA layers to FlashInfer's SM120 sparse
# MLA kernel (which accepts the per-head `sinks`) and HCA to the SM120 DSV4
# attention class. Upstream hard-wires FlashMLA, which rejects sm_121.
COPY attention_patched.py /opt/ernie/lib/python3.12/site-packages/vllm_ernie/models/ernie5_5/nvidia/attention.py
# b12x 1.5.1's SiTU MoE wrapper drops the W4A8 layout kwargs that its own
# dispatcher passes; forward them like the SiLU wrapper does.
COPY situ_patched.py /opt/ernie/lib/python3.12/site-packages/b12x/moe/_shared/kernels/situ.py

WORKDIR /workspace
