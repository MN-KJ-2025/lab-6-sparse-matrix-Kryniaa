# =================================  TESTY  ===================================
# Testy do tego pliku zostały podzielone na dwie kategorie:
#
#  1. `..._invalid_input`:
#     - Sprawdzające poprawną obsługę nieprawidłowych danych wejściowych.
#
#  2. `..._correct_solution`:
#     - Weryfikujące poprawność wyników dla prawidłowych danych wejściowych.
# =============================================================================
import numpy as np
import scipy as sp


def is_diagonally_dominant(A: np.ndarray | sp.sparse.csc_array) -> bool | None:
    if not isinstance(A, (np.ndarray, sp.sparse.csc_array)):
        return None
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return None
    diagonals =np.abs(A.diagonal())
    var = np.sum(np.abs(A),axis=1) - diagonals
    if(np.all(var<diagonals)):
        return True
    else:
        return False


def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float | None:
    if not isinstance(A, np.ndarray) or not isinstance(x, np.ndarray) or not isinstance(b, np.ndarray):
        return None
    if A.ndim != 2 or x.ndim != 1 or b.ndim != 1:
        return None
    if A.shape[0] != b.shape[0] or A.shape[1] != x.shape[0]:
        return None
    r = A @ x - b
    return np.linalg.norm(r)
