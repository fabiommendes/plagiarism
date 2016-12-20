def fibo(n):
    if n <= 2:
        return 1
    else:
        L = [1, 1]
        while len(L) < n:
            L.append(L[-1] + L[-2])
    return L[-1]