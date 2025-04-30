REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .on_policy_runner import OnPolicyRunner
REGISTRY["on_policy"] = OnPolicyRunner
