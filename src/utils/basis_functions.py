from src.basis_functions.basis_functions import PolynomialBasisFunction, RBFBasisFunction, SigmoidBasisFunction, RBFBasisFunctionMultidim


BASIS_FUNCTIONS_REGISTRY = {
    "PolynomialBasisFunction": PolynomialBasisFunction,
    "RBFBasisFunction": RBFBasisFunction,
    "SigmoidBasisFunction": SigmoidBasisFunction,
    "RBFBasisFunctionMultidim": RBFBasisFunctionMultidim,
}
