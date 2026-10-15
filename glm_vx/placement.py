"""Where routed experts run: per-layer choice between the primary backend and a host backend.

M4's hybrid CPU+GPU MoE: attention, dense MLPs, shared experts, routers, norms and the
output head always use the model's primary backend (the GPU); the routed experts of
`host_layers` run on `host_backend` instead (the CPU path), reading weights from host
memory. `from_verdict` builds the split an `admission.analyze` verdict chose.
"""


class ExpertPlacement:
    def __init__(self, host_backend, host_layers):
        self.host_backend = host_backend
        self.host_layers = frozenset(int(i) for i in host_layers)

    def backend_for(self, layer, primary):
        return self.host_backend if layer in self.host_layers else primary

    @classmethod
    def from_verdict(cls, verdict, config, host_backend):
        """First `expert_layers_on_gpu` MoE layers stay on the primary backend; the rest go to the host."""
        moe = [i for i, kind in enumerate(config.get('mlp_layer_types') or []) if kind == 'sparse']
        if len(moe) != verdict.moe_layers:
            raise ValueError('verdict was computed for a different model (MoE layer count differs)')
        return cls(host_backend, moe[verdict.expert_layers_on_gpu:])
