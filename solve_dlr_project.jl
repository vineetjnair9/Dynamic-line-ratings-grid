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


result["solution"]["branch"]


