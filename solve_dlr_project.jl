### Using PowerModels v0.18.4
### Note run_dc_opf is named differently in latest version

using PowerModels, Gurobi
using DataFrames, CSV

Path_Vineet = "/Users/vinee/ScenarioData/texas.m"
# Path_Thomas = "D:/papers/texas_test.m" 
result = run_dc_opf(Path_Vineet,
                    Gurobi.Optimizer,
                    setting = Dict("output" => Dict("duals" => true)))
;


d = result["solution"]["bus"]


lmps = [[parse(Int,k), d[k]["lam_kcl_r"]] for k in keys(d)];
lmps = DataFrame(Tables.table(transpose(hcat(lmps...))));
# CSV.write("D:/papers/lmps.csv", lmps)
CSV.write("/Users/vinee/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/MIT/Semesters/Spring 2022/15.S08/Project/lmps.csv", lmps)

g = result["solution"]["gen"]

gens = [[parse(Int,k), g[k]["pg"]] for k in keys(g)];
gens = DataFrame(Tables.table(transpose(hcat(gens...))));
# CSV.write("D:/papers/gens.csv", gens)
CSV.write("/Users/vinee/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/MIT/Semesters/Spring 2022/15.S08/Project/gens.csv", gens)


result["solution"]["branch"]


