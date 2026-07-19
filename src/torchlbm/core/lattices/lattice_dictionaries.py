from torchlbm.core.lattices.d1q2 import D1Q2
from torchlbm.core.lattices.d1q3 import D1Q3
from torchlbm.core.lattices.d2q9 import D2Q9
from torchlbm.core.lattices.d2q9_tensor import D2Q9Tensor
from torchlbm.core.lattices.d2q16 import D2Q16
from torchlbm.core.lattices.d3q19 import D3Q19

OneDimensionalLattices = {
    "D1Q2": D1Q2(),
    "D1Q3": D1Q3(),
}

TwoDimensionalLattices = {"D2Q9": D2Q9(), "D2Q9T": D2Q9Tensor(), "D2Q16": D2Q16()}

ThreeDimensionalLattices = {"D3Q19": D3Q19()}
