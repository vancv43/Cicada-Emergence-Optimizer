# RC22 to RC33, official CEC2020 code

RC22 to RC33 are evaluated with the C++ port of the official CEC2020 real-world constrained code by V. Stanovov (`cec2020rwc.cpp`, GPL-3), which needs Eigen 3.4.0. Because of its license the source is not part of this MIT repository. Obtain it from the competition organizers, github.com/P-N-Suganthan, place `cec2020rwc.cpp` and `eigen-3.4.0/` in `experiments/rwx/cec2020rwc_src/`, and build

```bash
g++ -O2 -std=c++14 -shared -fPIC -I experiments/rwx/cec2020rwc_src/eigen-3.4.0 \
    experiments/rwx/cec2020rwc_src/cec2020rwc.cpp -o experiments/rwx/cec2020rwc.so
```

`rw_official.py` loads `cec2020rwc.so` from this folder or from the path in `CEC2020RWC_SO`. The saved runs in `runs/rwx/`, `runs/rwa/` and `runs/rwb/` are enough to recompute every table without building it.
