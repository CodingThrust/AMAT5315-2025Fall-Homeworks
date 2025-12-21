# Homework 8
1.
- ij,kj->ik
- ij->
- ij,ij,ij->ij
- ij,kl,mn-> ijklmn

2.
Using the following code:
```julia
using OMEinsum

tn = ein"abc,cde,ehg,gfb->adhf"
optcontraction = optimize_code(tn, uniformsize(tn, 2), TreeSA())
```
The output is:
```julia
optcontraction = optimize_code(tn, uniformsize(tn, 2), TreeSA())
SlicedEinsum{Char, DynamicNestedEinsum{Char}}
(Char[], cdhg, acgf -> adhf
├─ cde, ehg -> cdhg
│  ├─ cde
│  └─ ehg
└─ abc, gfb -> acgf
   ├─ abc
   └─ gfb
)
```

3.
I use the code hw8_2.jl, and the result is:
```julia
[1.8066109403976796e18, 
6.875665762146549e18, 
6.151122581721419e19, 
1.2302123108278568e21, 
5.144558716182041e22, 
4.1116282893989705e24, 
5.612702683289912e26, 
1.162248872680485e29, 
3.285330233500941e31, 
1.1667408970547076e34, 
4.898099153436165e36, 
2.327394879239758e39, 
1.213677663873264e42, 
6.793958994977612e44, 
4.017291439227067e47, 
2.4794022967126456e50, 
1.5828842349953103e53, 
1.0380999176571235e56, 
6.95642903063234e58, 
4.7430829554740556e61]
```