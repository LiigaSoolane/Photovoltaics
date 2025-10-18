import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
from scipy.signal import find_peaks
import numpy as np
from scipy.optimize import curve_fit


#mpl.rcParams.update({
#    "text.usetex": True,
#    "pgf.texsystem": "lualatex",  # zwingt lualatex
#    "pgf.preamble": r"\usepackage{siunitx}"
#})

def reffunc(index, m, d):
    return (2*d/(m - index))


def ex1():
    
    df = pd.read_csv("data/Reflection.csv", sep=",", header=[0,1], decimal=".")
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
    plt.savefig("build/reflections.pdf") 

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

    reflection = (pd.to_numeric(df.iloc[:, 1], errors='coerce')/ 100)**0.5
    refractive_index = (1 + reflection) / (1 - reflection)

    index = np.linspace(0,9, 10)
    print(index)
    y = x_col[peaks] / refractive_index[peaks]
    
    m = 9.5
    best_d = None
    best_error = np.inf

    def model_d(x, d):
        return reffunc(x, 10, d)

    popt, _ = curve_fit(model_d, index, y)
    best_d = popt[0]

    d = (m-index)/refractive_index[peaks] * x_col[peaks] / 2
    print(np.mean(d))

    
    fig1, ax3 = plt.subplots(1, 1) 
    plt.plot(x_col[peaks], best_d*2*refractive_index[peaks]/(m-index), label="fit_best")
    plt.plot(x_col[peaks], np.mean(d)*2*refractive_index[peaks]/(m-index), label="fit_mean")
    plt.plot(x_col[peaks], x_col[peaks], label="data")
    plt.savefig("build/fit.pdf")
    print("Optimal m:", 10)
    print("Optimal d:", best_d)

    
    
    



def ex2():
    # read the data
    
    a_dark = np.genfromtxt("data/UI_a-Si_dark.txt", skip_header=4)
    a_light = np.genfromtxt("data/UI_a-Si_light.txt",  skip_header=4)
    c_dark = np.genfromtxt("data/UI_c-Si_dark.txt",  skip_header=4)
    c_light = np.genfromtxt("data/UI_c-Si_light.txt", skip_header=4)
    
    U_a_dark = a_dark[:, 0]
    A_a_dark = a_dark[:, 1]
    U_a_light = a_light[:, 0]
    A_a_light = a_light[:, 1]

    U_c_dark = c_dark[:, 0]
    A_c_dark = c_dark[:, 1]
    U_c_light = c_light[:, 0]
    A_c_light = c_light[:, 1]

    #plot the IU curve  
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 2, 1)
    ax2 = fig1.add_subplot(1, 2, 2)

    ax1.plot(U_a_dark, A_a_dark, label="dark")
    ax1.plot(U_a_light, A_a_light, label="light")
    ax1.set_title("amorphous Silicon")
    ax1.set_xlabel(r"Voltage $[\si{\volt}]$")
    ax1.set_ylabel(r"Current $[\si{\milli\ampere}]$")
    ax1.legend(loc="upper left")


    ax2.plot(U_c_dark, A_c_dark, label="dark")
    ax2.plot(U_c_light, A_c_light, label="light")
    ax2.set_title("crystalline Silicon")
    ax2.set_xlabel(r"Voltage $[\si{\volt}]$")
    ax2.set_ylabel(r"Current $[\si{\milli\ampere}]$")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig("build/UI.pdf")

    #a) find R_S and R_P



    #b) find the V_oc and I_sc
    V_oc_index_a = np.argmin(np.abs(A_a_light)) 
    I_sc_index_a = np.argmin(np.abs(U_a_light))
    V_oc_index_c = np.argmin(np.abs(A_c_light)) 
    I_sc_index_c = np.argmin(np.abs(U_c_light))

    print("V_oc amorphous:", U_a_light[V_oc_index_a])
    print("I_sc amorphous:", A_a_light[I_sc_index_a])
    print("V_oc crystalline:", U_c_light[V_oc_index_c])
    print("I_sc crystalline:", A_c_light[I_sc_index_c])

    #c) calculate the FF

    P_a = U_a_light[I_sc_index_a:V_oc_index_a] * A_a_light[I_sc_index_a:V_oc_index_a]
    P_c = U_c_light[I_sc_index_c:V_oc_index_c] * A_c_light[I_sc_index_c:V_oc_index_c]
    
    P_a_max = np.max(np.abs(P_a))
    P_c_max = np.max(np.abs(P_c))

    FF_a = P_a_max / (A_a_light[I_sc_index_a] * U_a_light[V_oc_index_a])
    FF_c = P_c_max / (A_c_light[I_sc_index_c] * U_c_light[V_oc_index_c])

    print("FF amorphous:", FF_a)
    print("FF crystalline:", FF_c)

    #d) find the efficiencies

    solar_irradiation_d = 1000 # W/m^2

    eta_a = P_a_max / (0.004**2 *solar_irradiation_d)
    eta_c = P_c_max / (0.004**2 *solar_irradiation_d)

    print("Efficiency amorphous:", eta_a)
    print("Efficiency crystalline:", eta_c)


ex1()
ex2()
