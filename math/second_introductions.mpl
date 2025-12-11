# Invasion-like scenario with multiple introductions
# m is the unidirectional migration rate from the source population
# Nec is the current effective population size of the focal population
# Nef is the founder effective population size of the focal population
# Nea is the current effective population size of the source population
# T1 and T2 are the time intervals for different epochs
# u is the distance between loci
with(LinearAlgebra):
# Epoch 1
S1 := Matrix([
    [-2*m - 1/(2*Nec),    0,             2*m,       1/(2*Nec),   0], # CC
    [0,             -1/(2*Nea),  0,        0,           1/(2*Nea)], # AA
    [0,           m,       -m, 0,          0],               # CA
    [0,               0,          0,            0,           0],               # C_coal (absorbing)
    [0,               0,          0,            0,           0]                # A_coal (absorbing)
]);

S2 := Matrix([
    [-2*m - 1/(2*Nef),    0,             2*m,       1/(2*Nef),   0], # CC
    [0,             -1/(2*Nea),  0,        0,           1/(2*Nea)], # AA
    [0,           m,       -m, 0,          0],               # CA
    [0,               0,          0,            0,           0],               # C_coal (absorbing)
    [0,               0,          0,            0,           0]                # A_coal (absorbing)
]);

# Init probabilities.
# We assume all samples are from the focal population
alpha := Transpose(Vector([1, 0, 0, 0, 0]));  # row vector
# Lineages might coalesce either in the focal population or the source population
exit  := Vector([0, 0, 0, 1, 1]);             # column vector

# Cumulative distribution for epoch 1 and epoch 2
cdf1 := alpha . MatrixExponential(t*S1) . exit;
cdf2 := alpha . MatrixExponential(T1*S1) . MatrixExponential((t - T1)*S2) . exit;

# The third epoch is a single panmintic population with no migration
# At time T2, compute the probability vector of states
state_at_T2 := alpha . MatrixExponential(T1*S1) . MatrixExponential((T2-T1)*S2);
# For epoch 3, only non-coalesced lineages continue
# After T2, all merge into ancestral_2, both lineages coalesce at rate 1/(2*Nea)
prob_not_coalesced_by_T2 := state_at_T2[1] + state_at_T2[2] + state_at_T2[3];
cdf3 := 1 - prob_not_coalesced_by_T2 * exp(-1/(2*Nea) * (t - T2));

# Simplify
cdf1_simp := simplify(cdf1);
cdf2_simp := simplify(cdf2);
cdf3_simp := simplify(cdf3);

# It would be nice to find a closed-form expression. That is, later, we want to evaluate:
# S(u) = int_0^inf exp(-2*t*u) * f(t) dt
# that in terms of the CDF it becomes
E1 := 2*u*int(exp(-2*u*t)*cdf1_simp, t=0..T1);
E2 := 2*u*int(exp(-2*u*t)*cdf2_simp, t=T1..T2);
E3 := 2*u*int(exp(-2*u*t)*cdf3_simp, t=T2..infinity);

S_u := simplify(E1+E2+E3);
S_u_simplified := simplify(S_u) assuming t>0, T1>0, u>0, Nec>0, Nea>0, Nef>0, T2>0, T2 > T1;
with(CodeGeneration):
Python(S_u_simplified);
# The final step is to compute the binned expectation
# S_bin = int_{u_i}^{u_j} S_u_simplified du / (u_j - u_j)
S_bin := simplify(int(S_u_simplified, u=u_i..u_j) / (u_j - u_i)) assuming u_i>0, u_j>u_i, t>0, T1>0, u>0, Nec>0, Nea>0, Nef>0, T2>0, T2 > T1;
Python(S_bin);
8 * (-(Nea * Nec * m - Nec / 4 + Nea / 4) * (Ei(1, T2 * (4 * Nea * u_i + 1) / Nea / 2) - Ei(1, T2 * (4 * Nea * u_j + 1) / Nea / 2)) * Nef * (Nea * Nef * m + Nef ** 2 * m + Nea / 2 - Nef / 2) * (Nec * m + 0.1e1 / 0.2e1) * Nec * (Nea * m - 0.1e1 / 0.2e1) * math.exp((((-4 * Nef * T2 * m + T1 - T2) * Nec - Nef * T1) * Nea + T2 * Nec * Nef) / Nea / Nec / Nef / 2) - 2 * (Nec - Nef) * (Nea * Nec * m - Nec / 4 + Nea / 4) * (Ei(1, T2 * (4 * Nea * u_i + 1) / Nea / 2) - Ei(1, T2 * (4 * Nea * u_j + 1) / Nea / 2)) * Nef * (Nea * Nef * m + Nea / 4 - Nef / 4) * m * Nec * math.exp(((-2 * m * (T1 + T2) * Nec - T1) * Nea + T2 * Nec) / Nec / Nea / 2) + (Nec - Nef) * (Ei(1, T1 * (4 * Nea * u_i + 1) / Nea / 2) - Ei(1, T1 * (4 * Nea * u_j + 1) / Nea / 2)) * Nea ** 3 * (m * Nef + 0.1e1 / 0.2e1) * Nef * m ** 2 * (Nec * m + 0.1e1 / 0.2e1) * Nec * math.exp(-T1 * (4 * Nea * Nec * m + Nea - Nec) / Nec / Nea / 2) - 4 * Nea * (Nea * Nec * m - Nec / 4 + Nea / 4) * (Ei(1, 2 * T1 * (0.1e1 / 0.4e1 + (m + u_i) * Nef) / Nef) - Ei(1, 2 * T1 * (0.1e1 / 0.4e1 + (m + u_j) * Nef) / Nef) - Ei(1, 2 * T2 * (0.1e1 / 0.4e1 + (m + u_i) * Nef) / Nef) + Ei(1, 2 * T2 * (0.1e1 / 0.4e1 + (m + u_j) * Nef) / Nef)) * (m * Nef + 0.1e1 / 0.4e1) * (Nea * Nef * m + Nef ** 2 * m + Nea / 2 - Nef / 2) * (Nec * m + 0.1e1 / 0.2e1) * Nec * (Nea * m - 0.1e1 / 0.2e1) * math.exp((Nec - Nef) / Nef / Nec * T1 / 2) + 8 * Nef * (Nea * Nef * m + Nea / 4 - Nef / 4) * (Nea * (m * Nef + 0.1e1 / 0.2e1) * (Nec * m + 0.1e1 / 0.4e1) * ((Nec * m + 0.1e1 / 0.2e1) * Nea + Nec ** 2 * m - Nec / 2) * (Nea * m - 0.1e1 / 0.2e1) * Ei(1, 2 * T1 * (0.1e1 / 0.4e1 + (m + u_i) * Nec) / Nec) / 2 - Nea * (m * Nef + 0.1e1 / 0.2e1) * (Nec * m + 0.1e1 / 0.4e1) * ((Nec * m + 0.1e1 / 0.2e1) * Nea + Nec ** 2 * m - Nec / 2) * (Nea * m - 0.1e1 / 0.2e1) * Ei(1, 2 * T1 * (0.1e1 / 0.4e1 + (m + u_j) * Nec) / Nec) / 2 + math.exp(-T2 * (2 * Nea * m - 1) / Nea / 2) * (Nea * Nec * m - Nec / 4 + Nea / 4) * (m * Nef + 0.1e1 / 0.2e1) * m * Nec ** 2 * Ei(1, T2 * (4 * Nea * u_i + 1) / Nea / 2) / 2 - math.exp(-T2 * (2 * Nea * m - 1) / Nea / 2) * (Nea * Nec * m - Nec / 4 + Nea / 4) * (m * Nef + 0.1e1 / 0.2e1) * m * Nec ** 2 * Ei(1, T2 * (4 * Nea * u_j + 1) / Nea / 2) / 2 + (-(Nec - Nef) * (Nea * Nec * m - Nec / 4 + Nea / 4) * (Ei(1, T1 * (m + 2 * u_i)) - Ei(1, T1 * (m + 2 * u_j)) - Ei(1, T2 * (m + 2 * u_i)) + Ei(1, T2 * (m + 2 * u_j))) * m ** 2 * Nec * math.exp(-T1 * (2 * Nec * m + 1) / Nec / 2) / 2 + (m * Nef + 0.1e1 / 0.2e1) * ((Nec * m + 0.1e1 / 0.4e1) * ((Nec * m + 0.1e1 / 0.2e1) * Nea + Nec ** 2 * m - Nec / 2) * (Nea * m - 0.1e1 / 0.2e1) * math.log(1 + (4 * m + 4 * u_i) * Nec) / 2 - (Nec * m + 0.1e1 / 0.4e1) * ((Nec * m + 0.1e1 / 0.2e1) * Nea + Nec ** 2 * m - Nec / 2) * (Nea * m - 0.1e1 / 0.2e1) * math.log(1 + (4 * m + 4 * u_j) * Nec) / 2 + ((Nec / 4 - Nea * Nec * m - Nea / 4) * Ei(1, T2 * (m + 2 * u_i)) + (Nea * Nec * m - Nec / 4 + Nea / 4) * Ei(1, T2 * (m + 2 * u_j)) + Nea * (Nec * m + 0.1e1 / 0.2e1) * math.log(4 * Nea * u_i + 1) / 2 + Nea * (-Nec * m - 0.1e1 / 0.2e1) * math.log(4 * Nea * u_j + 1) / 2 + (math.log(m + 2 * u_j) - math.log(m + 2 * u_i)) * (Nea * Nec * m - Nec / 4 + Nea / 4)) * m ** 2 * Nec ** 2)) * Nea)) / Nec / Nea / Nef / (2 * Nea * m - 1) / (4 * Nea * Nec * m + Nea - Nec) / (2 * m * Nef + 1) / (4 * Nea * Nef * m + Nea - Nef) / (2 * Nec * m + 1) / (-u_j + u_i)

