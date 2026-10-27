# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
import os, numpy as np, cec2022_fast as C
z=np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'cec2022_data','reference_values.npz')); n=0
for D in (10,20):
    B=C.cec2022_benchmarks(D); got=np.array([[B[k][0](x) for x in z[f'P{D}']] for k in B])
    ref = z[f'R{D}']; n += got.size
    if np.array_equal(got, ref):
        print(f'D{D}: bitwise identical to official reference on {got.size} evaluations')
    else:
        rel = float(np.max(np.abs(got - ref) / np.maximum(1.0, np.abs(ref))))
        # platform/numpy-version floating differences are tolerated; anything larger is a real defect
        assert rel < 1e-10, f'MISMATCH D{D}: max rel diff {rel}'
        print(f'D{D}: not bitwise but max rel diff {rel:.2e} < 1e-10 vs official reference (platform numerics)')
