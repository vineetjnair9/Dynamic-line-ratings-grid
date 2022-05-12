### Using PowerModels v0.19.5
### Note run_dc_opf is named differently in latest version
##  LLsub ./submit_test.sh [1,8,1]

using PowerModels, Gurobi, Ipopt
using DataFrames, CSV
using JuMP, Ipopt
using PowerModelsAnnex
using Dates

<<<<<<< HEAD
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
=======
# Use the current-based ratings
# https://github.com/lanl-ansi/PowerModels.jl/issues/727
# MVA documentation: https://github.com/Breakthrough-Energy/PowerSimData/blob/f27ebcfc6823bfacb49e22e95b1e5d46432539b8/powersimdata/input/export_data.py

#Path = "/Users/vinee/ScenarioData/texas.m"
#output_folder = "/Users/vinee/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/MIT/Semesters/Spring 2022/15.S08/Project/"
input_folder = "input_cases_DLR/"  #D:/courses/15.s08/project
output_folder = "outputs_DLR/"     #D:/courses/15.s08/project/

dataloc = "input_cases_DLR/";
fnames = dataloc.*readdir(dataloc)

dts = Array(Dates.DateTime(2016,1,1,0):Dates.Hour(1):Dates.DateTime(2016,12,31,24));



# https://github.com/llsc-supercloud/teaching-examples/blob/master/Julia/word_count/JobArray/top5each.jl
task_id = parse(Int, ARGS[1])
num_tasks = parse(Int, ARGS[2])

# -------------------------------------------------------------
function helper(m)
    # Format ouptut files
    return DataFrame(Tables.table(transpose(hcat(m...))));
end;

gurobi_solver = JuMP.optimizer_with_attributes(Gurobi.Optimizer)

function solve_opf_file(dt)
    print(dt)
    # Path = "D:/papers/texas_test.m" 
    ###dt_str = fn[length(fn)-14:length(fn)-2]
    dt_str = Dates.format(dt, "yyyy-mm-dd_HH")
    Path = input_folder * "texas_" * dt_str * ".m"
    data = PowerModels.parse_file(Path)

    @time result = PowerModels.run_opf(data, DCPPowerModel, gurobi_solver, setting=Dict("output" => Dict("duals" => true)))

    d = result["solution"]["bus"]
    lmps = helper([[parse(Int,k), d[k]["va"], d[k]["lam_kcl_r"]] for k in keys(d)]);
    CSV.write(output_folder * "lmps_" * dt_str * ".csv", lmps)

    g = result["solution"]["gen"]
    gens = helper([[parse(Int,k), g[k]["pg"]] for k in keys(g)]);
    CSV.write(output_folder * "gens_" * dt_str * ".csv", gens)

    b = result["solution"]["branch"];
    branches = helper([[parse(Int,k), b[k]["pt"], b[k]["mu_sm_fr"], b[k]["mu_sm_to"]] for k in keys(b)]);
    CSV.write(output_folder * "branches_" * dt_str * ".csv", branches);
end;

# -------------------------------------------------------------

for i in task_id+1:num_tasks:length(dts)
    @time solve_opf_file(dts[i])
end


"""
using DelimitedFiles
julia> data = make_basic_network(parse_file("D:/courses/15.s08/project/input_cases/texas_2016-01-01_00.m"))
julia> ptdf = calc_basic_ptdf_matrix(data)

writedlm( "D:/courses/ptdf.csv",  ptdf, ',')
"""



"""
Old code
for (key, value) in data["branch"]
    if ! value["transformer"]
        value["c_rating_a"] = 1000
    end
end
#model = Model(Ipopt.Optimizer)
#set_optimizer_attribute(model, "NonConvex", 2)
#build_ac_opf(data, model)
#optimize!(model)
#ipopt_solver = JuMP.optimizer_with_attributes(Ipopt.Optimizer)

@time result_dc_cl = result = PowerModels._solve_opf_cl(data, DCPPowerModel, gurobi_solver,
                                                        setting = Dict("output" => Dict("duals" => true)));

#result_ac_cl = PowerModels._solve_opf_cl(data, ACPPowerModel, ipopt_solver,
#                          setting = Dict("output" => Dict("duals" => true)))
#PowerModels._solve_opf_cl(data, ACPPowerModel, Gurobi.Optimizer(NonConvex=2),
#                          setting = Dict("output" => Dict("duals" => true)))

#PowerModels._solve_opf_cl(data, ACRPowerModel, ipopt_solver,
#setting = Dict("output" => Dict("duals" => true)))

#result = solve_ac_opf(Path_Thomas, ipopt_solver,
#setting = Dict("output" => Dict("duals" => true)))
#;
# 84.968

#result = solve_dc_opf(Path_Thomas, Gurobi.Optimizer,
#setting = Dict("output" => Dict("duals" => true)))
#;

#result_dc = solve_dc_opf(Path_Thomas,
#                    Gurobi.Optimizer,
#                    setting = Dict("output" => Dict("duals" => true)))
#;
"""
>>>>>>> vineet
