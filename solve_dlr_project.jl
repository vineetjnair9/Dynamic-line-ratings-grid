### Using PowerModels v0.18.4
### Note run_dc_opf is named differently in latest version

using PowerModels, Gurobi, Ipopt
using DataFrames, CSV

# Calculate PTDF shift factors
data = make_basic_network(parse_file("D:/papers/texas_test.m"));

ptdf = calc_basic_ptdf_matrix(data);

# Solve DCOPF
result = run_dc_opf("D:/papers/texas_test.m",
                    Gurobi.Optimizer,
                    setting = Dict("output" => Dict("duals" => true)))
;

# Use this to get current constraint
# https://github.com/lanl-ansi/PowerModels.jl/issues/727

result = run_ac_opf(data,
                    Ipopt.Optimizer,
                    setting = Dict("output" => Dict("duals" => true)))
;


# Write results to CSV files
d = result["solution"]["bus"];
lmps = [[parse(Int,k), d[k]["lam_kcl_r"]] for k in keys(d)];
lmps = DataFrame(transpose(hcat(lmps...)));
CSV.write("D:/papers/lmps.csv", lmps);

g = result["solution"]["gen"];
gens = [[parse(Int,k), g[k]["pg"]] for k in keys(g)];
gens = DataFrame(transpose(hcat(gens...)));
CSV.write("D:/papers/gens.csv", gens);

b = result["solution"]["branch"]
mus = [[parse(Int,k), b[k]["mu_sm_to"], b[k]["mu_sm_fr"]] for k in keys(b)];
mus = DataFrame(transpose(hcat(mus...)));
CSV.write("D:/papers/mus.csv", mus);



##pt
