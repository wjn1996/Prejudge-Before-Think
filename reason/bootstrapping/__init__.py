from reason.bootstrapping.vanilla_sampler.cot import CoTSampler
from reason.bootstrapping.vanilla_sampler.io import IOSampler
from reason.bootstrapping.tree_sampler.tree_cot import TreeSampler

SAMPLERS = {
    "io": IOSampler,
    "vanilla": CoTSampler,
    "tree": TreeSampler,
}