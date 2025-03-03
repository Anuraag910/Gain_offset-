import numpy as np
import scipy.integrate as spi

# Constants
r0 = 2.82e-13  # Classical electron radius in cm
m_e_c2 = 511  # Electron rest energy in keV

def klein_nishina(E, theta):
    """Computes differential cross-section dσ/dΩ for a given photon energy E (keV) and scattering angle theta (radians)."""
    alpha = E / m_e_c2
    E_prime = E / (1 + alpha * (1 - np.cos(theta)))
    
    d_sigma_dOmega = (r0**2 / 2) * (E_prime / E)**2 * (
        (E_prime / E) + (E / E_prime) - np.sin(theta)**2
    )
    return d_sigma_dOmega  # cm²/sr

def total_kn_cross_section(E):
    """Computes total Klein-Nishina cross-section by integrating over all angles."""
    def integrand(theta):
        return klein_nishina(E, theta) * 2 * np.pi * np.sin(theta)
    
    sigma_total, _ = spi.quad(integrand, 0, np.pi)
    return sigma_total  # cm²

def scattering_probability(E, theta_min, theta_max):
    """Computes the fraction of scattered photons within a given angle range."""
    sigma_total = total_kn_cross_section(E)
    
    def integrand(theta):
        return klein_nishina(E, theta) *np.sqrt(2) * np.sin(theta) 
     #sqrt2 is taken in radian, angle made by horizontal detector to centre of vertical det
    
    sigma_range, _ = spi.quad(integrand, theta_min, theta_max)
    
    return sigma_range / sigma_total

# Example usage:
E = 356  # Incident photon energy in keV
theta_min = np.radians(50)  # Lower bound of angle range
theta_max = np.radians(80)  # Upper bound of angle range

prob = scattering_probability(E, theta_min, theta_max)
print(f"Fraction of scattered photons in range {np.degrees(theta_min)}° - {np.degrees(theta_max)}°: {prob:.3f}")
