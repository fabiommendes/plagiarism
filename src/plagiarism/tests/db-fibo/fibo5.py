def fibo(n):
    x, y = 1, 1
    for i in range(n):
        x, y = y, x + y
    return x
