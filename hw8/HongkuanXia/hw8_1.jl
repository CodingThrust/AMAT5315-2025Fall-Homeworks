using OMEinsum

tn = ein"abc,cde,ehg,gfb->adhf"
optcontraction = optimize_code(tn, uniformsize(tn, 2), TreeSA())