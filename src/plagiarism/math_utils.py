from math import sqrt


def cos_angle(u, v):
    """
    Return the cosine of the angle between two vectors.
    """

    return u.dot(v) / (norm_l2(u) * norm_l2(v))


def norm_l2(u):
    """
    Euclidean norm of vector u.
    """
    return sqrt((u * u).sum())


def norm_l1(u):
    """
    L1 norm for vector u.
    """
    return abs(u).sum()


def similarity(u, v, method=None, norm=None):
    """
    Return a normalized measure of similarity between two vectors.

    The resulting value is between 0 (no similarity) and 1 (identity).
    """

    norm = NORM_MAP.get(norm, norm)
    method = METHOD_MAP.get(method, method)

    if method == 'angle':
        return (cos_angle(u, v) + 1) / 2
    elif method == 'triangular':
        norm_u = norm(u)
        norm_v = norm(v)
        if norm_u == norm_v == 0:
            return 1.0
        return 1 - norm(u - v) / (norm_u + norm_v)
    else:
        raise ValueError('invalid similarity method: %r' % method)


METHOD_MAP = {
    None: 'triangular',
}

NORM_MAP = {
    None: norm_l1,
    'l1': norm_l1,
    'L1': norm_l1,
    'manhattan': norm_l1,
    'l2': norm_l2,
    'L2': norm_l2,
    'euclidean': norm_l2,
}