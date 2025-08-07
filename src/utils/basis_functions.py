from src.basis_functions.basis_functions import PolynomialBasisFunction, RBFBasisFunction


BASIS_FUNCTIONS_REGISTRY = {
    "PolynomialBasisFunction": PolynomialBasisFunction,
    "RBFBasisFunction": RBFBasisFunction,
}