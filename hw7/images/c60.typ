#import "@preview/cetz:0.2.2": canvas, draw, tree, plot
#set page(width: auto, height: auto, margin: 5pt)

#let norm(v) = calc.sqrt(v.map(x => calc.pow(x, 2)).sum())
#let distance(a, b) = norm(a.zip(b).map(x => x.at(0) - x.at(1)))
#let show-graph(vertices, edges, radius:0.2) = {
  import draw: *
  for (k, (i, j)) in vertices.enumerate() {
    circle((i, j), radius:radius, name: str(k), fill:white)
  }
  for (k, l) in edges {
    line(str(k), str(l))
  }
}

#let udg-graph(vertices, unit:1) = {
  let edges = ()
  for (k, va) in vertices.enumerate() {
    for (l, vb) in vertices.enumerate() {
      if l < k and distance(va, vb) <= unit {
        edges.push((k, l))
      }
    }
  }
  return edges
}



#let fullerene() = {
    let φ = (1+calc.sqrt(5))/2
    let res = ()
    for (x, y, z) in ((0.0, 1.0, 3 * φ), (1.0, 2 + φ, 2 * φ), (φ, 2.0, 2 * φ + 1.0)) {
		    for (α, β, γ) in ((x,y,z), (y,z,x), (z,x,y)) {
			      for loc in ((α,β,γ), (α,β,-γ), (α,-β,γ), (α,-β,-γ), (-α,β,γ), (-α,β,-γ), (-α,-β,γ), (-α,-β,-γ)) {
				        if not res.contains(loc) {
                    res.push(loc)
                }
            }
        }
    }
	  return res
}

#figure(canvas(length: 0.6cm, {
  import draw: *
  let s(it) = text(14pt, it)
  let vertices = fullerene()
  let edges = udg-graph(vertices, unit: calc.sqrt(5))
  show-graph(vertices.map(v=>(v.at(0), v.at(1)*calc.sqrt(1/5) + v.at(2)*calc.sqrt(4/5))), edges)
}))


