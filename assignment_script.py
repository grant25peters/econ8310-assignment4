import pandas as pd
import pymc as pm
import arviz as az

url = "https://raw.githubusercontent.com/dustywhite7/Econ8310/master/AssignmentData/cookie_cats.csv"
df = pd.read_csv(url)

df["retention_1"] = df["retention_1"].astype(int)
df["retention_7"] = df["retention_7"].astype(int)

gate30 = df[df["version"] == "gate_30"]
gate40 = df[df["version"] == "gate_40"]

y30_1 = gate30["retention_1"].values
y40_1 = gate40["retention_1"].values

with pm.Model() as model_1:
    p30 = pm.Beta("p30", alpha=1, beta=1)
    p40 = pm.Beta("p40", alpha=1, beta=1)

    obs30 = pm.Bernoulli("obs30", p=p30, observed=y30_1)
    obs40 = pm.Bernoulli("obs40", p=p40, observed=y40_1)

    diff = pm.Deterministic("diff", p40 - p30)

    trace_1 = pm.sample(2000, tune=1000, random_seed=123)

az.summary(trace_1, var_names=["p30", "p40", "diff"])

y30_7 = gate30["retention_7"].values
y40_7 = gate40["retention_7"].values

with pm.Model() as model_7:
    p30 = pm.Beta("p30", alpha=1, beta=1)
    p40 = pm.Beta("p40", alpha=1, beta=1)

    obs30 = pm.Bernoulli("obs30", p=p30, observed=y30_7)
    obs40 = pm.Bernoulli("obs40", p=p40, observed=y40_7)

    diff = pm.Deterministic("diff", p40 - p30)

    trace_7 = pm.sample(2000, tune=1000, random_seed=123)

az.summary(trace_7, var_names=["p30", "p40", "diff"])



print("1-day retention results")
print(az.summary(trace_1, var_names=["p30", "p40", "diff"]))

print("7-day retention results")
print(az.summary(trace_7, var_names=["p30", "p40", "diff"]))


diff_1 = trace_1.posterior["diff"].values.flatten()
diff_7 = trace_7.posterior["diff"].values.flatten()

print("P(gate_40 > gate_30) for 1-day:", (diff_1 > 0).mean())
print("P(gate_40 < gate_30) for 1-day:", (diff_1 < 0).mean())

print("P(gate_40 > gate_30) for 7-day:", (diff_7 > 0).mean())
print("P(gate_40 < gate_30) for 7-day:", (diff_7 < 0).mean())