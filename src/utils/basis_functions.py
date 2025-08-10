from src.basis_functions.basis_functions import PolynomialBasisFunction, RBFBasisFunction, SigmoidBasisFunction


BASIS_FUNCTIONS_REGISTRY = {
    "PolynomialBasisFunction": PolynomialBasisFunction,
    "RBFBasisFunction": RBFBasisFunction,
    "SigmoidBasisFunction": SigmoidBasisFunction,
}
