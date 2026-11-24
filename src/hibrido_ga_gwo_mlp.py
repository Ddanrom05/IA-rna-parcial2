import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------- 1. Carga ----------
df = pd.read_csv("star_classification.csv")
print("Registros originales:", len(df))

# ---------- 2. Limpieza: fila corrupta (-9999 en u, g, z) ----------
n_bad = (df["u"] < -1000).sum()
df = df[df["u"] > -1000].reset_index(drop=True)
print(f"Filas corruptas eliminadas: {n_bad} | Registros: {len(df)}")

# ---------- 3. Índices de color ----------
df["u_g"] = df["u"] - df["g"]
df["g_r"] = df["g"] - df["r"]
df["r_i"] = df["r"] - df["i"]
df["i_z"] = df["i"] - df["z"]

# ---------- 4. Descriptores y etiqueta ----------
features = ["u", "g", "r", "i", "z", "redshift", "u_g", "g_r", "r_i", "i_z"]
X, y = df[features], df["class"]
assert X.isna().sum().sum() == 0