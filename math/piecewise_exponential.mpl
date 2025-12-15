# Exponential piece
gamma1 := 1 / (2*(Nec * exp(-t*alpha)));
Gamma1 := int(1 / (2*(Nec * exp(-s*alpha))), s = 0..t);
tmrca1 := simplify(gamma1 * exp(-Gamma1));

gamma2 := 1 / (2*Nea);  # Coalescent rate
Gamma2 := subs(t = t0, Gamma1) + int(gamma2, s = t0..t);
tmrca2 := gamma2 * exp(-Gamma2);

piece1 := int(exp(-2*t*u) * tmrca1,  t = 0..t0);
piece2 := int(exp(-2*t*u) * tmrca2,  t = t0..infinity);

piece1_sim := simplify(piece1) assuming t>0, t0>0, u>0, Nec>0, Nea>0, alpha!=0;
piece2_sim := simplify(piece2) assuming t>0, t0>0, u>0, Nec>0, Nea>0, alpha!=0;

with(CodeGeneration):
Python(piece2_sim);
1 / (4 * u * Nea + 1) * math.exp(-(4 * u * alpha * Nec * t0 + math.exp(t0 * alpha) - 1) / alpha / Nec / 2)
# There is a singularity at alpha=0. We do taylor expansion there:
taylor_piece2_sim := convert(series(piece2_sim, alpha=0, 3), polynom);
Python(taylor_piece2_sim);
1 / (4 * u * Nea + 1) * math.exp(-t0 * (4 * Nec * u + 1) / Nec / 2) - 1 / (4 * u * Nea + 1) * math.exp(-t0 * (4 * Nec * u + 1) / Nec / 2) * t0 ** 2 / Nec * alpha / 4

# Non-closed piece1
Python(piece1_sim);
cg1 = 1 / Nec * sympy.integrate(math.exp((2 * t * alpha ** 2 * Nec - 4 * t * u * Nec * alpha - math.exp(t * alpha) + 1) / Nec / alpha / 2), t == xrange(0,t0)) / 2
# Piece1 when alpha=0
Python(simplify(convert(series(piece1_sim, alpha=0, 2), polynom)) assuming t>0, t0>0, u>0, Nec>0, Nea>0);
(-math.exp(-t0 * (4 * Nec * u + 1) / Nec / 2) + 1) / (4 * Nec * u + 1)

# With same TMRCA eq., we can derive expected TMRCA
E_tmrca1 := int(t * tmrca1,  t = 0..t0) assuming t0>0, Nec>0, Nea>0, alpha!=0;
E_tmrca2 := int(t * tmrca2,  t = t0..infinity) assuming t0>0, Nec>0, Nea>0, alpha!=0;
Python(simplify(E_tmrca2) assuming t>0, t0>0, u>0, Nec>0, Nea>0, alpha!=0);
cg1 = (2 * Nea + t0) * math.exp(-(-1 + math.exp(t0 * alpha)) / alpha / Nec / 2)
Python(simplify(t * tmrca1) assuming t>0, t0>0, u>0, Nec>0, Nea>0, alpha!=0);
cg2 = t / Nec * math.exp((2 * t * alpha ** 2 * Nec - math.exp(t * alpha) + 1) / alpha / Nec / 2) / 2
# Around 0
Python(simplify(convert(series(E_tmrca2, alpha=0, 2), polynom)) assuming t>0, t0>0, u>0, Nec>0, Nea>0, alpha!=0);
(2 * Nea + t0) * math.exp(-1 / Nec * t0 / 2)
Python(simplify(convert(series(t * tmrca1, alpha=0, 2), polynom)) assuming t>0, t0>0, u>0, Nec>0, Nea>0);
Python(convert(series(int(t * tmrca1,  t = 0..t0), alpha=0, 2), polynom) assuming t0>0, Nec>0, Nea>0);
-2 * math.exp(-0.1e1 / Nec * t0 / 2) * Nec - math.exp(-0.1e1 / Nec * t0 / 2) * t0 + 2 * Nec

