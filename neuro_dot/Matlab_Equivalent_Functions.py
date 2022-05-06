#General imports
import numpy as np

class matlab:


    def rms_py(rms_input):
        """
        For matrices (N x M), rms_py(rms_input) is a row vector containing the RMS value from each column
        
        For complex input, rms_py separates the real and imaginary portions of each column of rms_input, squares them,
        and then takes the square root of the mean of that value
        
        For real input, the root mean square is calculated as follows:
        """
        rms = []
        if rms_input.dtype == 'complex128': 
            for i in range(rms_input.shape[1]):
                a = rms_input[:,i]
                val = np.sqrt(np.mean(a.real**2 + a.imag**2))
                rms.append(val)

            rms = np.array(rms)
        else:
            for i in range(rms_input.shape[1]):
                a = rms_input[:,i]
                val = np.sqrt(np.mean(a**2))
                rms.append(val)
                
            rms = np.array(rms)

        return rms