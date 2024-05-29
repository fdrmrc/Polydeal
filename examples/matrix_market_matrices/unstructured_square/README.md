## Example: unstructured square 2D

Conventions:
- The coarsest level is indexed by $0$. 

- Matrix $P_l^{l+1}$ denotes the prolongation matrix from level (coarse) $l$ to level $l+1$ (finer):

- Rhs: vector of ones (not exported here)

- All matrices have been exported in the Matrix Market file format

Concerning the files in this folder `/ord2` (similarly for other folders):

`P_0_1`: prolongation from level 0 to 1

`P_1_2`: prolongation from level 1 to 2

`P_2_3`: prolongation from level 2 to 3

`P_3_4`: prolongation from level 3 to 4

`unstructured_square_ord_2_matrix`: matrix associated with the differential problem.
