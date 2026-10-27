# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""
State-of-the-art baselines for the CEC2022 comparison, taken from the mealpy
library (Van Thieu and Mirjalili, 2023) rather than reimplemented here, so the
SOTA side of the comparison uses published, independently maintained code.

Selected methods:
  L-SHADE          Tanabe and Fukunaga (2014), the standard strong DE baseline
  LSHADE-cnEpSin   Awad et al. (2017), CEC2017 competition-grade DE variant
  IMODE            Sallam et al. (2020), CEC2020 competition winner
  INFO             Ahmadianfar et al. (2022), weighted-mean-of-vectors method
  HBA              Hashim et al. (2022), recent swarm metaheuristic
  SCSO             Seyyedabbasi and Kiani (2023), recent swarm metaheuristic

Each is wrapped so it can be driven by the same FE-matched harness as our own
implementations: the wrapper exposes the signature
(func, lb, ub, dim, pop_size, max_iter, seed) and passes the counting objective
straight through, so the hard FE cap in fe_budget.py still applies. The
population sizes follow each method's own convention (18*D for the L-SHADE
family, which starts from that size and reduces it linearly; the harness
population otherwise).
"""
import logging
import warnings

import numpy as np

warnings.filterwarnings("ignore")
logging.getLogger("mealpy").setLevel(logging.ERROR)

from mealpy import FloatVar  # noqa: E402
from mealpy.evolutionary_based.SHADE import L_SHADE  # noqa: E402
from mealpy.sota_based.LSHADEcnEpSin import OriginalLSHADEcnEpSin  # noqa: E402
from mealpy.sota_based.IMODE import OriginalIMODE  # noqa: E402
from mealpy.math_based.INFO import OriginalINFO  # noqa: E402
from mealpy.swarm_based.HBA import OriginalHBA  # noqa: E402
from mealpy.swarm_based.SCSO import OriginalSCSO  # noqa: E402

# name -> (class, population size rule)
SOTA_SPECS = {
    "L-SHADE": (L_SHADE, lambda dim, pop: 18 * dim),
    "LSHADE-cnEpSin": (OriginalLSHADEcnEpSin, lambda dim, pop: 18 * dim),
    "IMODE": (OriginalIMODE, lambda dim, pop: max(pop, 4 * dim)),
    "INFO": (OriginalINFO, lambda dim, pop: pop),
    "HBA": (OriginalHBA, lambda dim, pop: pop),
    "SCSO": (OriginalSCSO, lambda dim, pop: pop),
}

# evaluations per epoch, in units of the population size, used by the harness
# to set each method's epoch count from the FE budget (measured, see
# probe_sota.py): the DE variants evaluate one trial vector per individual.
SOTA_COST = {"L-SHADE": 1.0, "LSHADE-cnEpSin": 1.0, "IMODE": 1.0,
             "INFO": 1.0, "HBA": 1.0, "SCSO": 1.0}


def make_sota(name):
    cls, pop_rule = SOTA_SPECS[name]

    def run(func, lb, ub, dim, pop_size, max_iter, seed=0):
        lb = np.asarray(lb, dtype=float) * np.ones(dim)
        ub = np.asarray(ub, dtype=float) * np.ones(dim)
        pop = int(pop_rule(dim, pop_size))
        # the harness sizes max_iter from pop_size; rescale to this method's pop
        epochs = max(int(max_iter * pop_size / pop), 2)
        problem = {
            "bounds": FloatVar(lb=lb.tolist(), ub=ub.tolist()),
            "obj_func": lambda x: float(np.asarray(func(np.asarray(x).reshape(1, -1))).ravel()[0]),
            "minmax": "min",
            "log_to": None,
            "save_population": False,
        }
        model = cls(epoch=epochs, pop_size=pop)
        model.solve(problem, seed=seed, mode="single")
        return dict(best_fitness=np.inf, curve=None)  # the harness reads the counter
    return run


SOTA_ALGORITHMS = {name: make_sota(name) for name in SOTA_SPECS}
