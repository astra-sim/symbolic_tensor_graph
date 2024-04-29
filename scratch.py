def generate_mappings(A, B, mapping=None, index=0, used=None):
    if mapping is None:
        mapping = {}
    if used is None:
        used = set()

    if index == len(A):
        yield mapping
        return

    for j in range(len(B)):
        if j not in used and A[index] == B[j]:
            used.add(j)
            mapping[index] = j
            yield from generate_mappings(A, B, mapping, index + 1, used)
            used.remove(j)
            del mapping[index]

# A = [2, 2, 2, 3, 7]
# B = [3, 2, 7, 2, 2]

# for mapping in generate_mappings(A, B):
    # print(mapping)


def prime_factors(n):
    factors = []
    # While n is divisible by 2, add 2 as a factor and divide n by 2
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    # Iterate from 3 to the square root of n, considering only odd numbers
    for i in range(3, int(n**0.5) + 1, 2):
        # While i divides n, add i as a factor and divide n by i
        while n % i == 0:
            factors.append(i)
            n //= i
    # If n is still greater than 2, it means it's a prime number greater than 2
    if n > 2:
        factors.append(n)
    return factors

# # Example usage:
# number = 308
# print("Prime factors of", number, "are:", prime_factors(number))



