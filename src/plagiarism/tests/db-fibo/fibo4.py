def fibo(n):
    i = 0
    x = 1
    y = 1
    while i < n:
        aux = x
        x = y
        y += aux
        i += 1
    return x
