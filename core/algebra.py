import numpy as np


def jclass_permutation(P):
    """Compute the J-class permutation for Z/PZ under multiplication.

    Elements are grouped by gcd(x, P). Returns indices that sort elements
    by J-class, plus metadata for plotting (boundaries, tick positions, labels).

    Args:
        P: the modulus

    Returns:
        dict with:
            sorted_indices: [P] array — indices that sort elements by J-class
            boundaries: array — cumulative boundaries between J-classes (for gridlines)
            tick_positions: array — center of each J-class block (for axis labels)
            tick_labels: list of str — e.g. ["J_1", "J_5", "J_7", ...]
            j_classes: [P] array — gcd(x, P) for each element
            unique_j: array — sorted unique J-class values
            class_counts: array — number of elements per J-class
    """
    elements = np.arange(P)
    j_classes = np.gcd(elements, P)
    sorted_indices = np.argsort(j_classes)
    sorted_j_classes = j_classes[sorted_indices]

    unique_j, class_counts = np.unique(sorted_j_classes, return_counts=True)
    boundaries = np.cumsum(class_counts)[:-1]
    tick_positions = np.concatenate(([0], boundaries)) + class_counts / 2 - 0.5
    tick_labels = [f"J_{j}" for j in unique_j]

    return {
        "sorted_indices": sorted_indices,
        "boundaries": boundaries,
        "tick_positions": tick_positions,
        "tick_labels": tick_labels,
        "j_classes": j_classes,
        "unique_j": unique_j,
        "class_counts": class_counts,
    }



def prime_factors(n):
    """Return distinct prime factors of n in ascending order."""
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def primitive_root_prime(p):
    """Find the smallest primitive root modulo prime p."""
    if p == 2:
        return 1

    phi = p - 1
    factors = prime_factors(phi)

    for g in range(2, p):
        if all(pow(g, phi // q, p) != 1 for q in factors):
            return g

    raise ValueError(f"No primitive root found for prime {p}")


def crt_axis_generator(modulus, active_prime):
    """Find a CRT axis generator for one cyclic factor of (Z/modZ)*.

    Returns an element x modulo `modulus` that is:
      - a primitive root mod active_prime
      - 1 mod every other prime factor of modulus

    Only valid when modulus is squarefree.
    """
    primes = prime_factors(modulus)
    g = primitive_root_prime(active_prime)

    for x in range(modulus):
        if x % active_prime != g:
            continue
        if all(x % p == 1 for p in primes if p != active_prime):
            return x

    raise ValueError(f"No CRT axis generator found for modulus {modulus}, prime {active_prime}")


def build_product_cycle(base, generators, orders, P):
    """Build an algebraically ordered orbit as a product of cyclic groups.

    Generates elements base * g1^k1 * g2^k2 * ... mod P,
    iterating each generator through its full order.

    Args:
        base: starting element (e.g. j for J-class j)
        generators: list of group generators mod P
        orders: list of orders for each generator
        P: modulus

    Returns:
        list of elements in product-cycle order
    """
    out = []

    def rec(curr, depth):
        if depth == len(generators):
            out.append(curr % P)
            return
        g = generators[depth]
        x = 1
        for _ in range(orders[depth]):
            rec((curr * x) % P, depth + 1)
            x = (x * g) % P

    rec(base % P, 0)
    return out


def get_jclass_elements(d, P):
    """Return all elements x in [0, P) where gcd(x, P) == d."""
    from math import gcd
    return [x for x in range(P) if gcd(x, P) == d]


def local_inverse_map(J, P):
    """Find the idempotent e_J and local inverse map within a J-class.

    Args:
        J: list of elements in the J-class (e.g. from get_jclass_elements)
        P: modulus

    Returns:
        (inv_map, e_J) where inv_map[c] = c# such that c * c# ≡ e_J (mod P)

    Raises:
        IndexError: if J contains no idempotent, or if any element has no inverse within J
    """
    e = [x for x in J if (x * x) % P == x][0]
    inv = {}
    for c in J:
        inv[c] = [x for x in J if (c * x) % P == e][0]
    return inv, e



def build_multi_log_table(j, P):
    """Build a multi-dimensional discrete log table for J_j.

    Maps each element z in J_j to an exponent tuple (t_1, ..., t_m)
    with the local identity e_J mapped to (0, ..., 0).

    Special case: j == P (the zero element) returns ({0: (0,)}, [0], [1], [1]).

    Returns:
        (log_table, generators, orders, primes)
    """
    J_j = get_jclass_elements(j, P)

    if j == P:
        return {0: (0,)}, [0], [1], [1]

    modulus = P // j
    primes = prime_factors(modulus)
    orders = [p - 1 for p in primes]
    generators = [crt_axis_generator(modulus, p) for p in primes]
    inv_Jj, e_Jj = local_inverse_map(J_j, P)

    log_table = {}

    def build_exp_tuple(depth, unit_mod, exp_tuple):
        if depth == len(generators):
            z = (e_Jj * unit_mod) % P
            log_table[z] = tuple(exp_tuple)
            return
        g = generators[depth]
        value = 1
        for t in range(orders[depth]):
            build_exp_tuple(depth + 1, (unit_mod * value) % modulus, exp_tuple + [t])
            value = (value * g) % modulus

    build_exp_tuple(0, 1, [])
    return log_table, generators, orders, primes


def character(z, freq_vec, orders, log_table):
    """Real character for a product of cyclic groups.

    Computes chi_k(z) = 2 * cos(sum_i 2*pi*k_i*t_i/o_i)
    where t_vec = log_table[z] is the exponent tuple.

    Args:
        z: element of a J-class (must be a key in log_table)
        freq_vec: frequency multi-index tuple, same length as orders
        orders: cyclic group orders — use the `orders` returned by build_multi_log_table
        log_table: exponent map — use the `log_table` returned by build_multi_log_table
    """
    import numpy as np
    t_vec = log_table[z]
    phase = sum(2.0 * np.pi * int(k) * int(t) / int(o)
                for k, t, o in zip(freq_vec, t_vec, orders))
    return 2.0 * np.cos(float(phase))


def algebraic_jclass_order(P):
    """Full algebraic J-class ordering for squarefree P.

    For each J-class J_j, orders elements using CRT decomposition of the
    unit group mod P/j. Returns the same dict format as jclass_permutation().

    Special case: j == P (the zero J-class, gcd(0, P) = P) is mapped to [0],
    since 0 is the unique zero-product element.

    Only valid when P is squarefree (product of distinct primes).

    Raises:
        ValueError: if P is not squarefree
    """
    # Verify squarefree
    primes_of_P = prime_factors(P)
    product = 1
    for p in primes_of_P:
        product *= p
    if product != P:
        raise ValueError(
            f"algebraic_jclass_order requires squarefree P, got {P} "
            f"(prime factorization includes repeated factors)"
        )

    elements = np.arange(P)
    j_classes = np.gcd(elements, P)
    unique_j = sorted(set(j_classes.tolist()))

    # CRT generators mod P for each prime factor
    generators_mod_P = [crt_axis_generator(P, p) for p in primes_of_P]
    orders_of_P = [p - 1 for p in primes_of_P]

    sorted_indices = []
    class_counts = []

    for j in unique_j:
        if j == P:
            indices = [0]
        else:
            # Active generators: those for primes NOT dividing j
            active_gens = []
            active_orders = []
            for p, g, o in zip(primes_of_P, generators_mod_P, orders_of_P):
                if j % p != 0:
                    active_gens.append(g)
                    active_orders.append(o)

            if active_gens:
                indices = build_product_cycle(j, active_gens, active_orders, P)
            else:
                indices = [j]

        sorted_indices.extend(indices)
        class_counts.append(len(indices))

    sorted_indices = np.array(sorted_indices)
    class_counts = np.array(class_counts)
    boundaries = np.cumsum(class_counts)[:-1]
    tick_positions = np.concatenate(([0], boundaries)) + class_counts / 2 - 0.5
    tick_labels = [f"J_{j}" for j in unique_j]

    return {
        "sorted_indices": sorted_indices,
        "boundaries": boundaries,
        "tick_positions": tick_positions,
        "tick_labels": tick_labels,
        "j_classes": j_classes,
        "unique_j": np.array(unique_j),
        "class_counts": class_counts,
    }
