import numpy as np
import pandas as pd
from google.colab import files

# Unggah file dataset
uploaded = files.upload()  # Ini akan membuka dialog untuk mengunggah file

# Pastikan nama file sesuai dengan yang diunggah
file_name = list(uploaded.keys())[0]  # Ambil nama file pertama yang diunggah
data = pd.read_csv(file_name)

# Define the function to find roots
# Example: Find a specific threshold in the Reviews column (e.g., f(x) = x - target_reviews)
def f(x, target_reviews):
    return x - target_reviews

# Bisection method implementation
def bisection_method(func, target, a, b, tol=1e-6, max_iter=100):
    """
    Bisection method to find the root of a function func within [a, b].

    Parameters:
    func : function - The function for which the root is sought.
    target : float - The target value to find (e.g., reviews).
    a, b : float - The interval within which to search.
    tol : float - The tolerance for convergence.
    max_iter : int - Maximum number of iterations.

    Returns:
    float - The root of the function if found, otherwise None.
    """
    fa = func(a, target)
    fb = func(b, target)

    if fa * fb > 0:
        raise ValueError("Function has the same sign at both endpoints. Choose a different interval.")

    for i in range(max_iter):
        c = (a + b) / 2
        fc = func(c, target)

        if np.abs(fc) < tol or (b - a) / 2 < tol:
            return c

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    raise ValueError("Maximum iterations exceeded without convergence.")

# Example usage
try:
    data['Reviews'] = pd.to_numeric(data['Reviews'], errors='coerce')  # Ubah Reviews jadi numerik
    cleaned_reviews = data['Reviews'].dropna().sort_values()          # Bersihkan data NaN dan urutkan

    # Tentukan target dan interval untuk bisection
    target_reviews = 50  # Target nilai dalam kolom Reviews
    a, b = cleaned_reviews.min(), cleaned_reviews.max()  # Gunakan min dan max dari data sebagai interval

    # Hitung akar menggunakan bisection method
    root = bisection_method(f, target_reviews, a, b)
    print(f"The root (Reviews = {target_reviews}) is approximately: {root}")

except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
    
