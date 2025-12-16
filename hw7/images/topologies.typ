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

#let triangles(n) = {
  let v = ()
  let e = ()
  for i in range(n){
    v.push((i, 0))
    v.push((i, 1))
    e.push((2*i, 2*i+1))
    if i > 0 {
      e.push((2*i - 2, 2*i))
      e.push((2*i - 1, 2*i))
      e.push((2*i - 1, 2*i+1))
    }
  }
  return (v, e)
}

#let squares(n) = {
  let v = ()
  let e = ()
  for i in range(n){
    v.push((i, 0))
    v.push((i, 1))
    e.push((2*i, 2*i+1))
    if i > 0 {
      e.push((2*i - 2, 2*i))
      e.push((2*i - 1, 2*i+1))
    }
  }
  return (v, e)
}

#let diamonds(n) = {
  let v = ()
  let e = ()
  v.push((0, 0))
  for i in range(n){
    v.push((1 * i + 0.5, 0.5))
    v.push((1 * i + 0.5, -0.5))
    v.push((1 * i + 1, 0))
    e.push((3*i, 3*i+1))
    e.push((3*i, 3*i+2))

    e.push((3*i+1, 3*i+3))
    e.push((3*i+2, 3*i+3))
  }
  return (v, e)
}


#canvas({
  import draw: *
  let (vs, es) = triangles(6)
  content((-0.1, 1.5), align(left)[Triangles])
  show-graph(vs, es)
  set-origin((0, -2))
  content((-0.2, 1.5), align(left)[Squares])
  let (vs, es) = squares(6)
  show-graph(vs, es)
  set-origin((0, -1.5))
  content((0, 1.0), align(left)[Diamonds])
  let (vs, es) = diamonds(5)
  show-graph(vs, es)
})