import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
from scipy.signal import find_peaks
import numpy as np
from scipy.optimize import curve_fit


mpl.rcParams.update({
    "text.usetex": True,
    "pgf.texsystem": "lualatex",  # zwingt lualatex
    "pgf.preamble": r"\usepackage{siunitx}"
})

def reffunc(index, m, d):
    return (2*d/(m - index))


def ex1():
    
    df = pd.read_csv("vXXX/data/Reflection.csv", sep=",", header=[0,1], decimal=".")
    #print(df.columns)
    num_cols = df.shape[1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 10)) 

    # Jede zweite Spalte ist x, die nächste y
    for i in range(0, num_cols-1, 2):
        x_col = df.columns[i]
        y_col = df.columns[i + 1]
        # Plotten, leere Werte ignorieren
        ax1.plot(df[x_col], df[y_col], label=df.columns[i][0])
        
        reflection = (df[y_col]/100)**(1/2)
        refractive_index = (1+reflection)/(1-reflection)
        ax2.plot(df[x_col], refractive_index, label=df.columns[i][0])


    # Achsenbeschriftungen, Legende etc.
    ax1.set_title("Reflection of several pv materials")
    ax1.set_ylabel(r"Reflection $[\si{\percent}]$")
    ax2.set_title("Refraction index of several pv materials")
    ax2.set_xlabel(r"wavelength $[\si{\angstrom}]$")
    ax2.set_ylabel(r"Refractive Index")
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    #plt.grid(True)
    plt.tight_layout()
    plt.savefig("vXXX/build/reflections.pdf") 

    x_col = pd.to_numeric(df.iloc[:, 4], errors='coerce')
    y_col = pd.to_numeric(df.iloc[:, 5], errors='coerce')

    peaks, properties = find_peaks(
        y_col,
        height=3,
        distance=30,
        prominence=0.5
    )

    peaks[4] = 1180

    print("x-values of maxima:", x_col.iloc[peaks])

    reflection = (y_col / 100)**0.5
    refractive_index = (1 + reflection) / (1 - reflection)

    index = np.linspace(0,9, 10)
    print(index)
    y = x_col.iloc[i] / refractive_index.iloc[i]
    
    best_m = None
    best_d = None
    best_error = np.inf

    for m_candidate in range(8, 12):
        # Fit only d for this integer m
        def model_d(x, d):
            return reffunc(x, m_candidate, d)

        popt, _ = curve_fit(model_d, index, y)
        d_candidate = popt[0]

        # Compute residuals
        residuals = y_data - model(x_data, m_candidate, d_candidate)
        error = np.sum(residuals**2)

        if error < best_error:
            best_error = error
            best_m = m_candidate
            best_d = d_candidate

    print("Optimal m:", best_m)
    print("Optimal d:", best_d)

    
    
    



def ex13():
    a = 1




ex1()
