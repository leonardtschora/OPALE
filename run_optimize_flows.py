import os
os.environ["OPALE"] = os.curdir
from scipy.optimize import linprog, least_squares
from src.optim.optim_problems import *

########################### 0) Load variables (20 s)
datesh, data, model, network, dims = load_data("train")
Pz, Pz_prime = load_prices(data, network, model, dims)
Cz, Rz, Gz = load_fundamentals(data, network, model, dims)
Vz = load_maximal_generation_capacities(datesh, network)
Az_zprime = load_network_constraints(data, network, model, dims)

########################### 1) Compute Flin (50 s)
# Load variables and formulate the problem
S, M, b, B = formulate_lin_problem(Pz, Pz_prime, Cz, Rz, Vz, Az_zprime,
                                   network, model, dims)
# Try the problem on 1 hour
h = 0
results = linprog(S[h, :], A_eq=M, b_eq=b[h, :], bounds=B[h, :, :].transpose())

# Compute Flin on the train set
Flin = compute_Flin(S, M, b, B, network, dims, datesh, data)

########################### 2) Compute Flsq (19 h), use Flsq = load("Flsq") to skip
f, x0, CD, B, b = formulate_lsq_problem(Pz, Pz_prime, Cz, Rz, Gz, Vz, Az_zprime,
                                        network, model, dims)

# Try the problem on 1 sample
h = 0
results = least_squares(f, x0, bounds=B[h], kwargs={"CD" : CD, "b" : b[h]})

# Compute Flsq on the train set
Flsq = compute_Flsq(f, x0, CD, B, b, network, (5, dims[1], dims[2]), datesh, data)
########################### 3) Compute Fcmb (160 s)
# Re-Load estimated Flows
F = load("F")
errors = compute_errors(F, Flin, Flsq, data[0])

# Inspect results
params = {"fontsize" : 45, "fontsize_labels" : 30, "linewidth" : 5} 
compare_flows(Flin, Flsq, F, "FR", "DE", ["Flin", "Flsq", "F"], data[0],
              dates_=[datetime.date(2019, 8, 12),
                      datetime.date(2019, 8, 18)], params=params) 
compare_flows(Flin, Flsq, F, "NONO5", "NONO1", ["Flin", "Flsq", "F"], data[0],
              dates_=[datetime.date(2018, 11, 28),
                      datetime.date(2018, 12, 4)], params=params) 

# Compute Ldiff and select (z, zprime, x, q)
rules = compute_rules(F, Flin, Flsq, data[0])

# Combine flows
Fcmb = compute_Fcmb(Flin, Flsq, rules, data[0])
########################### 4) Compute Funi (5 s)
# Identify one sided-flows
os_flows = one_sided_flows(F, Fcmb, data[0])

# Apply one-sideness
Funi = compute_Funi(Fcmb, os_flows)
########################### 5) Apply to the test set (10h). Uncomment to load
# Repeat the last 4 steps, but we use the found rules and os_flows
A,Flin_test,Flsq_test, Fcmb_test,Funi_test,dates=compute_test_set(F,rules,os_flows)
save_flows(Flin, Flsq, Fcmb, Funi, Flin_test, Flsq_test, Fcmb_test, Funi_test)

"""
datesh, data, model, network, dims = load_data("test")
dates = data[0]
F = load("F")
Flin_test = load("Flin")
Flsq_test = load("Flsq")
Fcmb_test = load("Fcmb")
Funi_test = load("Funi")
A = load("A")
"""
errors=compute_errors_test(A, F, Flin_test, Flsq_test, Fcmb_test, Funi_test, dates)
pvalues = compute_DM_tests(A, F, Flin_test, Flsq_test, Fcmb_test, Funi_test, dates)
