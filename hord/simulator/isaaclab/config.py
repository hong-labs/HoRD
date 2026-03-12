from dataclasses import dataclass, field
from hord.simulator.base_simulator.config import SimParams, SimulatorConfig
from hord.simulator.isaacgym.config import IsaacGymPhysXParams


@dataclass
class IsaacLabPhysXParams(IsaacGymPhysXParams):
    """PhysX physics engine parameters."""
    gpu_found_lost_pairs_capacity: int = 2**21
    gpu_max_rigid_contact_count: int = 2**23
    gpu_found_lost_aggregate_pairs_capacity: int = 2**25
    # Maximum number of rigid contact patches for the GPU pipeline.
    # Increase this to resolve "Patch buffer overflow" errors from PhysX.
    gpu_max_rigid_patch_count: int = 2**18


@dataclass
class IsaacLabSimParams(SimParams):
    """PhysX-specific simulation parameters used by IsaacGym and IsaacLab."""
    physx: IsaacLabPhysXParams =  field(default_factory=IsaacLabPhysXParams)


@dataclass
class IsaacLabSimulatorConfig(SimulatorConfig):
    """Configuration specific to IsaacLab simulator."""
    sim: IsaacLabSimParams  # Override sim type
    def __post_init__(self):
        self.w_last = False  # IsaacLab uses wxyz quaternions
