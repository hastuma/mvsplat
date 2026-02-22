from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor


from typing import Optional

@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    scales: Optional[Float[Tensor, "batch gaussian 3"]] = None
    rotations: Optional[Float[Tensor, "batch gaussian 4"]] = None
