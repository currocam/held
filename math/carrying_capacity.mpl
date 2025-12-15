# Constant piece (carrying capacity)
gamma1 := 1 / (2*Nec);
Gamma1 := int(gamma1, s = 0..t);
tmrca1 := simplify(gamma1 * exp(-Gamma1));

# Exponential piece
gamma2 := 1 / (2*(Nec * exp(-(t-t0)*alpha)));
Gamma2 := subs(t = t0, Gamma1) + int( 1 / (2*(Nec * exp(-(s-t0)*alpha))), s = t0..t);
tmrca2 := gamma2 * exp(-Gamma2);
# Constant piece
gamma3 :=1 / (2*Nea);
Gamma3 := subs(t = t1, Gamma2) + int(gamma3, s = t1..t);
tmrca3 := gamma3 * exp(-Gamma3);

# Pieces
piece1 := int(exp(-2*t*u) * tmrca1,  t = 0..t0);
piece2 := int(exp(-2*t*u) * tmrca2,  t = t0..t1);
piece3 := int(exp(-2*t*u) * tmrca3,  t = t1..infinity);

piece1_sim := simplify(piece1) assuming t>0, t0>0, t1>0, u>0, Nec>0, Nea>0, alpha!=0;
piece2_sim := simplify(piece2) assuming t>0, t0>0, t1>0, u>0, Nec>0, Nea>0, alpha!=0;
piece3_sim := simplify(piece3) assuming t>0, t0>0, t1>0, u>0, Nec>0, Nea>0, alpha!=0;

S_u_simplified := simplify(piece1_sim + piece2_sim + piece3_sim);
with(CodeGeneration):
Python(piece1_sim);
cg = (1 - math.exp(-t0 * (4 * u * Nec + 1) / Nec / 2)) / (4 * u * Nec + 1)
Python(piece3_sim);
cg0 = math.exp((1 - math.exp(-(t0 - t1) * alpha) - (4 * Nec * t1 * u + t0) * alpha) / alpha / Nec / 2) / (4 * u * Nea + 1)
s_ut2 := simplify(exp(-2*t*u) * tmrca2) assuming t>t0, t0>0, t1>0, u>0, Nec>0, Nea>0, alpha!=0;
 Python(s_ut2);
cg1 = 1 / Nec * math.exp((-math.exp((t - t0) * alpha) + 1 + (2 * t - 2 * t0) * Nec * alpha ** 2 + (-4 * Nec * t * u - t0) * alpha) / alpha / Nec / 2) / 2

# There's a singularity at alpha=0
taylor_piece3_sim := simplify(convert(series(piece3_sim, alpha=0, 2), polynom));
Python(taylor_piece3_sim);
cg2 = math.exp(-t1 * (4 * u * Nec + 1) / Nec / 2) / (4 * u * Nea + 1)
# Also:
taylor_s_ut2 := simplify(convert(series(s_ut2, alpha=0, 2), polynom));
Python(taylor_s_ut2);
cg3 = 1 / Nec * math.exp(-t * (4 * u * Nec + 1) / Nec / 2) / 2

# With same TMRCA eq., we can derive expected TMRCA
E_tmrca := int(t * tmrca1,  t = 0..t0) + int(t * tmrca2,  t = t0..t1) + int(t * tmrca3,  t = t1..infinity);
E_tmrca_sim := simplify(E_tmrca) assuming t0>0, t1>t0, Nec>0, Nea>0, alpha!=0;
Python(E_tmrca_sim);
(sympy.integrate(t * math.exp((-math.exp((t - t0) * alpha) + 1 + (2 * t - 2 * t0) * Nec * alpha ** 2 - t0 * alpha) / alpha / Nec / 2), t == xrange(t0,t1)) + (2 * t1 + 4 * Nea) * Nec * math.exp(-(t0 * alpha + math.exp(-(t0 - t1) * alpha) - 1) / alpha / Nec / 2) + (-4 * Nec ** 2 - 2 * Nec * t0) * math.exp(-1 / Nec * t0 / 2) + 4 * Nec ** 2) / Nec / 2
# Approaching to zero
Python(simplify(convert(series(E_tmrca_sim, alpha=0, 2), polynom)));
(2 * Nea - 2 * Nec) * math.exp(-1 / Nec * t1 / 2) + 2 * Nec
