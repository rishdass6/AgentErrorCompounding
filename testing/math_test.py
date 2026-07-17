a = [2, 5, 17, 68]

for n in range(4, 31):
    val = (
        9*a[n-1]
        -30*a[n-2]
        +44*a[n-3]
        -24*a[n-4]
        + (n*n - 3*n + 2) * (2**n)
    )
    a.append(val)

print(a[30])