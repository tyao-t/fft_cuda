import cmath
import string
import random

def generate_random_integers(num_integers, max_num_digits):
    random_integers = []
    for _ in range(num_integers):
        digits = string.digits
        num_digits = random.randint(1, max_num_digits)
        random_digits = ''.join(random.choice(digits) for _ in range(num_digits))
        random_integer = int(random_digits)
        random_integers.append(random_integer)
    return random_integers

def normalize(result):
    n = len(result)
    return [int(round(x.real / n)) for x in result]

def multiply_polynomials(poly1, poly2):
    n = 1
    while n < len(poly1) + len(poly2):
        n *= 2

    poly1.extend([0] * (n - len(poly1)))
    poly2.extend([0] * (n - len(poly2)))
    fft_poly1 = fast_dft(poly1)
    fft_poly2 = fast_dft(poly2)
    fft_product = [a * b for a, b in zip(fft_poly1, fft_poly2)]
    product_poly = normalize(fast_dft(fft_product, inverse=True))

    # Remove leading zeros
    while len(product_poly) > 1 and product_poly[-1] == 0:
        product_poly.pop()

    return product_poly

def multiply_large_numbers(num1, num2):
    poly1 = [int(digit) for digit in str(num1)]
    poly2 = [int(digit) for digit in str(num2)]

    product_poly = multiply_polynomials(poly1, poly2)
    product_num = 0
    for digit in product_poly:
        product_num = product_num * 10 + digit

    return product_num

def nth_roots_of_unity(n, conjugate=False):
    roots = []
    for k in range(n):
        root = cmath.exp((2j if conjugate else -2j) * cmath.pi * k/n)
        roots.append(root)
    return roots

def fast_dft(arr, inverse=False):
    if len(arr) == 1:
        return [arr[0]]
    eve = fast_dft(arr[::2], inverse)
    odd = fast_dft(arr[1::2], inverse)

    m = len(arr)
    res = [0] * m
    omega = nth_roots_of_unity(m, inverse)
    for i in range(m // 2):
        res[i] = eve[i] + omega[i]*odd[i]
        res[i + m // 2] = eve[i] - omega[i]*odd[i]
    return res

def main():
    num1, num2 = generate_random_integers(2, 4300)
    final_ans = multiply_large_numbers(num1, num2)
    print(f"Product of {num1} and {num2} is {final_ans}")
    assert(final_ans == num1 * num2)

if __name__ == "__main__":
    main()