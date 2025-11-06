# Task 1
My package link: https://github.com/Jazztempo/MyFirstPackage.jl

# Task 2
For 
```
  fib(n) = n <= 2 ? 1 : fib(n - 1) + fib(n - 2)
```
time complexity is $\Theta((\frac{1+\sqrt{5}}{2})^n)$

For 
```
  function fib_while(n)
      a, b = 1, 1
      for i in 3:n
          a, b = b, a + b
      end
      return b
  end
```
time complexity is O(n)
