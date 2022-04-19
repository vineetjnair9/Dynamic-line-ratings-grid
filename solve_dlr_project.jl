### Using PowerModels v0.18.4
### Note run_dc_opf is named differently in latest version

using PowerModels, Gurobi
using DataFrames, CSV


result = run_dc_opf("D:/papers/texas_test.m",
                    Gurobi.Optimizer,
                    setting = Dict("output" => Dict("duals" => true)))
;


d = result["solution"]["bus"]


lmps = [[parse(Int,k), d[k]["lam_kcl_r"]] for k in keys(d)];
lmps = DataFrame(transpose(hcat(lmps...)));
CSV.write("D:/papers/lmps.csv", lmps)


g = result["solution"]["gen"]

gens = [[parse(Int,k), g[k]["pg"]] for k in keys(g)];
gens = DataFrame(transpose(hcat(gens...)));
CSV.write("D:/papers/gens.csv", gens)



result["solution"]["branch"]


