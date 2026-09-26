# -*- coding: utf-8 -*-
"""
CEC2014 benchmark wrapper for DMGSO.

Requires:
    pip install opfunu

Provides a unified interface:
    problem.objective(x)
    problem.lower_bound
    problem.upper_bound
    problem.dimension
    problem.function_id
    problem.optimum
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np


@dataclass
class CEC2014Problem:
    function_id: int
    dimension: int
    objective: Callable[[np.ndarray], float]
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    optimum: float
    name: str
    category: str


def cec2014_category(function_id: int) -> str:
    if 1 <= function_id <= 3:
        return "Unimodal"
    if 4 <= function_id <= 16:
        return "Multimodal"
    if 17 <= function_id <= 22:
        return "Hybrid"
    if 23 <= function_id <= 30:
        return "Composition"
    raise ValueError(f"Invalid CEC2014 function ID: {function_id}")


def _load_cec_function(function_id: int, dimension: int):
    """
    Load CEC2014 function from opfunu.

    Compatible with opfunu 1.x structure:
        opfunu.cec_based.cec2014
    """

    try:
        from opfunu.cec_based.cec2014 import (
            F12014, F22014, F32014, F42014, F52014,
            F62014, F72014, F82014, F92014, F102014,
            F112014, F122014, F132014, F142014, F152014,
            F162014, F172014, F182014, F192014, F202014,
            F212014, F222014, F232014, F242014, F252014,
            F262014, F272014, F282014, F292014, F302014,
        )
    except ImportError as exc:
        raise ImportError(
            "\n[ERROR] CEC2014 functions could not be imported from opfunu.\n"
            "Please make sure opfunu is installed in the active environment:\n\n"
            "    python -m pip install opfunu\n"
        ) from exc

    funcs = {
        1: F12014,
        2: F22014,
        3: F32014,
        4: F42014,
        5: F52014,
        6: F62014,
        7: F72014,
        8: F82014,
        9: F92014,
        10: F102014,
        11: F112014,
        12: F122014,
        13: F132014,
        14: F142014,
        15: F152014,
        16: F162014,
        17: F172014,
        18: F182014,
        19: F192014,
        20: F202014,
        21: F212014,
        22: F222014,
        23: F232014,
        24: F242014,
        25: F252014,
        26: F262014,
        27: F272014,
        28: F282014,
        29: F292014,
        30: F302014,
    }

    if function_id not in funcs:
        raise ValueError("CEC2014 function_id must be between 1 and 30.")

    return funcs[function_id](ndim=dimension)


def get_cec2014_problem(function_id: int, dimension: int = 30) -> CEC2014Problem:
    """
    Return a standardized CEC2014 problem object.

    Parameters
    ----------
    function_id:
        CEC2014 function ID, from 1 to 30.
    dimension:
        Problem dimension. For the main CEC2014 evaluation, D=30 is commonly used.
    """

    if function_id < 1 or function_id > 30:
        raise ValueError("CEC2014 function_id must be between 1 and 30.")

    func = _load_cec_function(function_id, dimension)

    # Most CEC2014 functions use [-100, 100]^D.
    lower = getattr(func, "lb", -100.0)
    upper = getattr(func, "ub", 100.0)

    if np.isscalar(lower):
        lb = np.full(dimension, float(lower), dtype=float)
    else:
        lb = np.asarray(lower, dtype=float)

    if np.isscalar(upper):
        ub = np.full(dimension, float(upper), dtype=float)
    else:
        ub = np.asarray(upper, dtype=float)

    optimum = float(getattr(func, "f_global", 0.0))

    def objective(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(func.evaluate(x))

    return CEC2014Problem(
        function_id=function_id,
        dimension=dimension,
        objective=objective,
        lower_bound=lb,
        upper_bound=ub,
        optimum=optimum,
        name=f"CEC2014-F{function_id}",
        category=cec2014_category(function_id),
    )


def list_cec2014_functions() -> List[int]:
    return list(range(1, 31))


if __name__ == "__main__":
    problem = get_cec2014_problem(function_id=1, dimension=30)
    x0 = np.zeros(problem.dimension)

    print(problem.name)
    print("Category:", problem.category)
    print("D:", problem.dimension)
    print("Lower bound:", problem.lower_bound[0])
    print("Upper bound:", problem.upper_bound[0])
    print("f(x0):", problem.objective(x0))
    print("f_opt:", problem.optimum)