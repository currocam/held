# Exponential piece
gamma1 := Nec * exp(-t*alpha);
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
1 / (4 * u * Nea + 1) * math.exp(-(4 * u * Nec * alpha * t0 + math.exp(t0 * alpha) - 1) / Nec / alpha / 2)
# There is a singularity at alpha=0. We do taylor expansion there:
taylor_piece2_sim := convert(series(piece2_sim, alpha=0, 3), polynom);
Python(taylor_piece2_sim);
1 / (4 * u * Nea + 1) * math.exp(-t0 * (4 * Nec * u + 1) / Nec / 2) - 1 / (4 * u * Nea + 1) * math.exp(-t0 * (4 * Nec * u + 1) / Nec / 2) * t0 ** 2 / Nec * alpha / 4

