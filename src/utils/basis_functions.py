from src.basis_functions.basis_functions import (PolynomialBasisFunction,
                                                 RBFBasisFunction,
                                                 SigmoidBasisFunction,
                                                 RBFBasisFunctionMultidim,
                                                 SigmoidBasisFunctionMultidim,
                                                 PolynomialBasisFunctionMultidim
                                                 )


BASIS_FUNCTIONS_REGISTRY = {
    "PolynomialBasisFunction": PolynomialBasisFunction,
    "RBFBasisFunction": RBFBasisFunction,
    "SigmoidBasisFunction": SigmoidBasisFunction,
    "RBFBasisFunctionMultidim": RBFBasisFunctionMultidim,
    "SigmoidBasisFunctionMultidim": SigmoidBasisFunctionMultidim,
    "PolynomialBasisFunctionMultidim": PolynomialBasisFunctionMultidim,
}
