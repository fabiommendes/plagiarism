def fibo(n):
    x, y = 1, 1
    for _ in range(n):
        x, y = y, x + y
    return x