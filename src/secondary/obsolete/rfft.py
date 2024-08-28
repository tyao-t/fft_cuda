import cmath

def fft(poly):
    n = len(poly)
    if n <= 1:
        return poly
    even = fft(poly[0::2])
    odd = fft(poly[1::2])
    t = [cmath.exp(-2j * cmath.pi * k / n) * odd[k] for k in range(n // 2)]
    return [even[k] + t[k] for k in range(n // 2)] + [even[k] - t[k] for k in range(n // 2)]

def ifft(poly):
    n = len(poly)
    if n <= 1:
        return poly
    even = ifft(poly[0::2])
    odd = ifft(poly[1::2])
    t = [cmath.exp(2j * cmath.pi * k / n) * odd[k] for k in range(n // 2)]
    return [even[k] + t[k] for k in range(n // 2)] + [even[k] - t[k] for k in range(n // 2)]

def normalize_ifft(result):
    n = len(result)
    result = ifft(result)
    return [int(round(x.real / n)) for x in result]

def multiply_polynomials(poly1, poly2):
    n = 1
    while n < len(poly1) + len(poly2):
        n *= 2
    print(n)
    poly1.extend([0] * (n - len(poly1)))
    poly2.extend([0] * (n - len(poly2)))
    fft_poly1 = fft(poly1)
    fft_poly2 = fft(poly2)
    print(fft_poly1)
    print(fft_poly2)
    fft_product = [a * b for a, b in zip(fft_poly1, fft_poly2)]
    print(fft_product)
    product_poly = normalize_ifft(fft_product)

    # Remove leading zeros
    while len(product_poly) > 1 and product_poly[-1] == 0:
        product_poly.pop()

    return product_poly

def multiply_large_numbers(num1, num2):
    # Convert numbers to polynomials
    poly1 = [int(digit) for digit in str(num1)]
    poly2 = [int(digit) for digit in str(num2)]

    # Multiply polynomials
    product_poly = multiply_polynomials(poly1, poly2)
    # Convert polynomial back to number
    product_num = 0
    for digit in product_poly:
        product_num = product_num * 10 + digit

    print(product_num)
    return product_num

# num1 = 1234
# num2 = 9876
# result = multiply_large_numbers(num1, num2)
x = fft([6, 7, 8, 9, 0, 0, 0, 0])
y = fft([4, 3, 2, 1, 0, 0, 0, 0])
print(x)
print(y)
print([a*b for a,b in zip(x, y)])
import numpy as np
print(np.fft.ifft([a*b for a,b in zip(x, y)]))
# product_poly = ifft([a*b for a,b in zip(x, y)])
# product_num = 0
# for digit in product_poly:
#     product_num = product_num * 10 + digit

# print(product_num)
# print(123456789*987654321)
# print(result)  # Output: 121932631112635269