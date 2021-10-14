df(name) = CSV.File(joinpath(ROOT, "data", "$name.csv")) |> DataFrame
for s in DATASETS
    str = string(s)
    f_ex = string("load_", str) |> Symbol
    quote
        $f_ex() = df($str)
    end |> eval
end
