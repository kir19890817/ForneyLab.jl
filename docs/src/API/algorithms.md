# ForneyLab
```@index
Modules = [ForneyLab]
```

```@autodocs
Modules = [ForneyLab]
Private = false
Pages = collect(Iterators.flatten([[joinpath(root[4:end], file) for file in files] for (root, dirs, files) in walkdir("../src/algorithms/")]))
```