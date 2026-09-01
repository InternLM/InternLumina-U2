from torch.distributed._tensor import Shard

from veomni.distributed.parallel_plan import ParallelPlan


def get_parallel_plan():
    ep_plan = {
        "model.layers.*.mlp.experts.gate_proj": Shard(0),
        "model.layers.*.mlp.experts.up_proj": Shard(0),
        "model.layers.*.mlp.experts.down_proj": Shard(0),
        # Split export: experts is a ModuleList and each projection is a weight.
        "model.layers.*.mlp.experts.*.gate_proj.weight": Shard(0),
        "model.layers.*.mlp.experts.*.up_proj.weight": Shard(0),
        "model.layers.*.mlp.experts.*.down_proj.weight": Shard(0),
    }
    parallel_plan = ParallelPlan(
        ep_plan=ep_plan,
    )
    return parallel_plan
