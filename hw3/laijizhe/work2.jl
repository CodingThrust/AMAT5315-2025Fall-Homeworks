# For fib(n) = n <= 2?1 : fib(n −1) + fib(n −2)
# is exponential due to redundant recursive calls forming a binary recursion
# tree. Each call branches into two, leading to approximately O(φn) growth, where
# φ = (1+√5) / 2≈1.618 is the golden ratio. In Big-O notation, this is commonly
# simplified and bounded above by:
# O(2^n)


# For the second one, it is linear, as it performs a single loop from 3 to n,
# executing constant-time operations in each iteration. Thus, the time complexity
# is:
# O(n)
