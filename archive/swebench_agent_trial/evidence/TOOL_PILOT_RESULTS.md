# Original C3-S versus FastCtx versus D3-S: five-task pilot

This is a preliminary real-model pilot, not a powered benchmark claim. All 15
agent runs completed successfully and all 15 patches were evaluated by the
official SWE-bench Verified Docker harness.

| Arm | Resolved | Tokens / task | Latency / task | Repository search |
|---|---:|---:|---:|---|
| C3-S (original Codex) | 4/5 | 412,038 | 95.8s | shell: 3.8 calls, 10,290 chars |
| C3-SF (Codex + FastCtx) | 4/5 | 373,847 | 90.4s | FastCtx: 14.4 calls, 21,384 chars |
| D3-S | 4/5 | 203,589 | 36.2s | FS plugin: 6.8 calls, 12,761 chars |

All three arms reached 80% accuracy on exactly the same tasks. Relative to the
original C3-S, C3-SF used 9.3% fewer tokens and 5.7% less agent wall time;
D3-S used 50.6% fewer tokens and 62.2% less agent wall time. Relative to C3-SF,
D3-S used 45.5% fewer tokens and 59.9% less wall time.

The corrected treatment was followed: original C3-S used its normal shell
search, C3-SF used FastCtx `grep`/`glob`, and D3-S used its filesystem plugin.
C3-SF and D3-S made zero shell `rg`/`grep`/`find` calls. FastCtx did not simply
reduce search volume: C3-SF made more search calls and received more search
characters than original C3-S. Its modest total-token reduction therefore
cannot be attributed to smaller raw search output alone. D3-S still accumulated
substantially less cached input than either Codex arm (177,050 versus 369,382
for C3-S and 336,461 for C3-SF).

None of the arms invoked the technical-document KB on these five tasks.
Therefore, this pilot tests filesystem-search and harness orchestration, not KB
retrieval accuracy or KB coexistence. Five tasks are also too few for a general
claim. Original C3-S was run after the alternating C3-SF/D3-S pairs rather than
interleaved with them, so temporal or service-state effects remain possible.
The appropriate conclusion is only that FastCtx alone did not erase D3-S's
efficiency advantage in this small sample.

Artifacts:

- Raw run state and trajectories: `results/full_agent_fastctx_d3s_v2/`
- Original C3-S run state and trajectories: `results/full_agent_fastctx_pilot5_c3s_v1/`
- Official reports: `results/full_agent_fastctx_d3s_v2/official_pilot5/`
- Original C3-S official report: `results/full_agent_fastctx_pilot5_c3s_v1/official_pilot5/`
- Fixed pilot manifest: `evaluation/manifests/swebench_fastctx_pilot5_v1.json`
- Corrected treatment protocol: `evaluation/full_agent_fastctx_d3s_protocol_v2.json`
