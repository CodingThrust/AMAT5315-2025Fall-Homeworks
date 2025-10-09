# Homework 3 解答（huichengzhang）

## 1. Package Creation（提交仓库链接）

请按照课程指引完成，并在此粘贴你的 GitHub 仓库链接（示例占位）：

仓库链接：https://github.com/zhc1212/MyFirstPackage.jl

完成要点清单（核对用）：
- 已使用 `PkgTemplates.jl` 或手动创建 `MyFirstPackage.jl`
- 已添加 `Project.toml`、`src/MyFirstPackage.jl`、`test/runtests.jl`
- 已配置 CI：
  - GitHub Actions（`/.github/workflows/ci.yml`）包含：
    - 多版本 Julia（例如 1.8, 1.9, 1.10）
    - 安装依赖、运行测试、生成覆盖率
  - Codecov 上传覆盖率（可选）
- 测试全部通过、覆盖率≥80%
- 未执行 General 注册步骤

参考的最简 CI 示例（粘贴到 `/.github/workflows/ci.yml`）：

```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        julia-version: [ '1.8', '1.9', '1.10' ]
    steps:
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v2
        with:
          version: ${{ matrix.julia-version }}
      - uses: julia-actions/cache@v2
      - uses: julia-actions/julia-buildpkg@v1
      - uses: julia-actions/julia-runtest@v1
      - uses: julia-actions/julia-processcoverage@v1
      - uses: codecov/codecov-action@v4
        with:
          files: lcov.info
```

`src/MyFirstPackage.jl` 样例骨架：

```julia
module MyFirstPackage

export greet, fib

greet(name::AbstractString) = "Hello, $(name)!"

function fib(n::Integer)
    n < 0 && throw(ArgumentError("n must be nonnegative"))
    n <= 2 && return 1
    a, b = 1, 1
    @inbounds for _ in 3:n
        a, b = b, a + b
    end
    return b
end

end # module
```

`test/runtests.jl` 样例：

```julia
using Test
using MyFirstPackage

@test greet("World") == "Hello, World!"
@test fib(1) == 1
@test fib(2) == 1
@test fib(10) == 55
```

## 2. Big-O Analysis

题目给出：

```julia
fib(n) = n <= 2 ? 1 : fib(n - 1) + fib(n - 2)
```

- 时间复杂度：$\Theta\left(\left(\frac{1+\sqrt{5}}{2}\right)^n\right)$，即精确的黄金比例φ的n次方。常用上界写作 O(2^n)，指数级。
- 空间复杂度：O(n)（递归栈深度）。

迭代版本：

```julia
function fib_while(n)
    a, b = 1, 1
    for i in 3:n
        a, b = b, a + b
    end
    return b
end
```

- 时间复杂度：O(n)
- 空间复杂度：O(1)


