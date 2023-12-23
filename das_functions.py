import numpy as np
import scipy.optimize as optimize
import scipy.integrate as integrate
import scipy.fft as fft
import scipy.signal as signal
import os

def read_data(data_dir,channel_no=69,time_samples_per_file=25000):
    """
    Function to consolidate raw distributed acoustic sensing (DAS) output files into 
    a numpy array for analysis. 
    
    param: data_dir: path to directory containing raw DAS output files for given experiment. 
    param: channel_no: number of channels / sensors along the pipe
    param: time_samples_per_file: number of time samples in each DAS output file
    returns: data: 2D-array of DAS phase displacements [-π,π] at each sensor over time.
                   Each row corresponds to a specific point in time, while each column 
                   corresponds to a specific channel / sensor. 
    """
    data_list = []
    for filename in np.sort(os.listdir(data_dir)):
        filepath = os.path.join(data_dir,filename)
        with open(filepath, 'rb') as f:
            file_data = np.fromfile(f, dtype=np.uint16)
        file_data_array = file_data.reshape((channel_no, time_samples_per_file))
        file_data_array = file_data_array*(2*np.pi)/65535-np.pi
        data_list.append(file_data_array)
    
    data = np.hstack(data_list).T
    return(data)

def transform_to_fk_domain(data,sampling_frequency,spatial_resolution,workers=None):
    """
    Transform raw DAS timeseries data into the frequency-wavenumber (f-k) domain. 
    Returns a 2-D power spectrum (periodogram). 
    
    param: data: 2D-array of DAS phase displacements [-π,π] at each sensor over time.
    param: sampling_frequency: Frequency of time samples [Hz]
    param: spatial_resolution: Distance between each sensor [m]
    param: workers: number of workers to use in parallel computation
    
    returns: P: Power spectrum of f-k pairs
    returns: f: frequency values [1/s] associated with rows of A. 
    returns: k: wavenumber values [1/m] associated with columns of A. 
    """
    # Frequencies
    nt=len(data[:,0])
    dt = 1/sampling_frequency
    nyq_f = nt//2
    f = fft.fftfreq(nt, d=dt)[0:nyq_f]
    
    # Wavenumbers
    nx = len(data[0,:])
    nyq_k = nx//2
    k = fft.fftshift(fft.fftfreq(nx, d=spatial_resolution))
    
    # Frequency-wavenumber fourier coefficients
    coeffs = fft.fftshift(fft.fft2(data,workers=workers)[0:nyq_f,:],axes=1)
    
    # Square magnitude of fourier coefficients to get estimate of power spectrum
    P = np.abs(coeffs)**2
    return(P,f,k)

def transform_to_fk_domain_welch(data,sampling_frequency,spatial_resolution,segment_length,workers=None):
    """
    Transform raw DAS timeseries data into the frequency-wavenumber (f-k) domain using Welch's method.  
    This method computes an estimate of the power spectrum by dividing the data into overlapping segments, 
    computing a periodogram for each segment, and averaging the periodograms.
    
    Using this method helps to reduce noise compared to the standard periodogram calculation, but at the 
    expense of frequency bin resolution. 
    
    param: data: 2D-array of DAS phase displacements [-π,π] at each sensor over time.
    param: sampling_frequency: Frequency of time samples [Hz]
    param: spatial_resolution: Distance between each sensor [m]
    param: segment_length: number of timepoints to use per segment. 
    param: workers: number of workers to use in parallel computation
    
    returns: P: Power spectrum of f-k pairs
    returns: f: frequency values [1/s] associated with rows of A. 
    returns: k: wavenumber values [1/m] associated with columns of A. 
    """
    data_length = data.shape[0]
    half_segment = int(segment_length//2)
    segment_length = int(half_segment*2)
    P = np.zeros((half_segment,data.shape[1]))
    
    segment_start = 0
    segment_end = segment_length
    
    num_segments = 0
    
    while segment_end < data_length:
        segment = slice(segment_start,segment_end,1)
        P_segment,f,k = transform_to_fk_domain(data[segment],sampling_frequency,spatial_resolution,workers=workers)
        P += P_segment
        num_segments += 1
        segment_start += half_segment
        segment_end += half_segment
        
    if num_segments > 0:
        P = P/num_segments
    else:
        P,f,k = transform_to_fk_domain(data,sampling_frequency,spatial_resolution,workers=workers)
        
    print(f'Number of segments: {num_segments}')
    
    return(P,f,k)

def line_integral_downstream(P,f,k,slope):
    """
    Calculate the line integral of power spectrum returned by 2D discete fourier transform 
    over the slope line associated with a specified downstream (i.e., positive slope) speed of sound.
    
    Algorithm is based on the one described in Appendix A of Ahmed Abukhamsin's PhD thesis
    (url: http://purl.stanford.edu/db484pp7251). 
    
    param: P: Power spectrum (f-k plot). Rows correspond to frequency, columns to wavenumber.  
    param: f: frequency values [1/s] associated with rows of A. 
    param: k: wavenumber values [1/m] associated with columns of A. 
    param: slope: speed of sound guess over which to integrate [m/s]. 
    returns: integral_value: value of line integral, normalized by length of line. 
    """
    
    # Get spacing between f-k plot grid cells 
    dk = k[1] - k[0]
    df = f[1] - f[0]
    
    # Start at origin
    i = np.argmin(np.abs(f))
    j = np.argmin(np.abs(k))
    
    # Specify stopping criteria 
    imax = len(f)
    jmax = len(k)
    
    # Initialize position in f-k space
    ft = f[i]
    kt = k[j]
    
    # Pre-compute values that we'll use frequently inside loop
    hk = dk/2
    hf = df/2
    theta = np.arctan(slope)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Take line integral of power spectrum.
    # The contribution of each cell to integral is equivalent to it's value
    # multiplied by the lenght of the line segment that is inside the cell. 

    L_sum = 0
    LP_sum = 0

    while (i < imax) and (j < jmax):

        L1 = (k[j] + hk - kt)/cos_theta
        L2 = (f[i] + hf - ft)/sin_theta

        if L1 < L2:
            L = L1
            L_sum += L
            LP_sum += L*P[i,j]
            kt = k[j] + hk
            ft = slope*kt
            j += 1
        else:
            L = L2
            L_sum += L
            LP_sum += L*P[i,j]
            ft = f[i] + hf
            kt = ft/slope
            i += 1

    integral_value = LP_sum/L_sum    
    
    return(integral_value,L_sum)

def line_integral_upstream(P,f,k,slope):
    """
    Calculate the line integral of power spectrum returned by 2D discete fourier transform 
    over the slope line associated with a specified upstream (i.e., negative slope) speed of sound.
    
    Algorithm is based on the one described in Appendix A of Ahmed Abukhamsin's PhD thesis
    (url: http://purl.stanford.edu/db484pp7251). 
    
    param: P: Power spectrum (f-k plot). Rows correspond to frequency, columns to wavenumber. 
    param: f: frequency values [1/s] associated with rows of A. 
    param: k: wavenumber values [1/m] associated with columns of A. 
    param: slope: speed of sound guess over which to integrate [m/s]. 
    returns: integral_value: value of line integral, normalized by length of line. 
    """
    
    # Get spacing between f-k plot grid cells 
    dk = k[1] - k[0]
    df = f[1] - f[0]
    
    # Start at origin
    i = np.argmin(np.abs(f))
    j = np.argmin(np.abs(k))
    
    # Specify stopping criteria
    imax = len(f)
    jmin = 0
    
    # Initialize position in f-k space
    ft = f[i]
    kt = k[j]
    
    # Pre-compute values that we'll use frequently inside loop
    hk = dk/2
    hf = df/2
    theta = np.arctan(-1*slope)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Take line integral of power spectrum. 
    # The contribution of each cell to integral is equivalent to it's value
    # multiplied by the lenght of the line segment that is inside the cell.

    L_sum = 0
    LP_sum = 0

    while (i < imax) and (j > jmin):

        L1 = (kt - k[j] + hk)/cos_theta
        L2 = (f[i] + hf - ft)/sin_theta

        if L1 < L2:
            L = L1
            L_sum += L
            LP_sum += L*P[i,j]
            kt = k[j] - hk
            ft = slope*kt
            j -= 1
        else:
            L = L2
            L_sum += L
            LP_sum += L*P[i,j]
            ft = f[i] + hf
            kt = ft/slope
            i += 1

    integral_value = LP_sum/L_sum    
    
    return(integral_value,L_sum)

def doppler_objective_function(x,P,f,k):
    """
    Objective function that is minimized when estimating flow velocity (u) and speed of sound (c) in pipeline.
    The goal here is to maximize the line integral of the power spectrum over the 
    lines formed by upstream / downstream sound waves. 
    
    param: x: list consisting of flow velocity and speed of sound guesses [u,c]
    param: P: Power spectrum (f-k plot). Rows correspond to frequency, columns to wavenumber. 
    param: f: frequency values [1/s] associated with rows of A. 
    param: k: wavenumber values [1/m] associated with columns of A. 
    """
    u,c = x
    c_downstream = u + c
    c_upstream = u - c
    
    downstream_integral,downstream_L = line_integral_downstream(P,f,k,c_downstream)
    upstream_integral,upstream_L = line_integral_upstream(P,f,k,c_upstream)
    combined_integral = (downstream_L*downstream_integral + upstream_L*upstream_integral)/(downstream_L + upstream_L)
    objective_function_value = -1*combined_integral
    
    return(objective_function_value)

def vector_line_integral_downstream(P,f,k,slope):
    """
    Calculate the line integral of power spectrum returned by 2D discete fourier transform 
    over the slope line associated with a specified downstream (i.e., positive slope) speed of sound.
    
    Return a vector with value of power spectrum at different lengths along line. 
    
    Algorithm is based on the one described in Appendix A of Ahmed Abukhamsin's PhD thesis
    (url: http://purl.stanford.edu/db484pp7251). 
    
    param: P: Power spectrum (f-k plot). Rows correspond to frequency, columns to wavenumber.  
    param: f: frequency values [1/s] associated with rows of A. 
    param: k: wavenumber values [1/m] associated with columns of A. 
    param: slope: speed of sound guess over which to integrate [m/s]. 
    returns: integral_value: value of line integral, normalized by length of line. 
    """
    
    # Get spacing between f-k plot grid cells 
    dk = k[1] - k[0]
    df = f[1] - f[0]
    
    # Start at origin
    i = np.argmin(np.abs(f))
    j = np.argmin(np.abs(k))
    
    # Specify stopping criteria 
    imax = len(f)
    jmax = len(k)
    
    # Determine maximum number of grid cells that line could potentially pass through
    # Multiply by factor of 1.25 to give ourselves extra room
    nmax = int(1.25*np.sqrt((imax-i)**2 + (jmax-j)**2))
    L_vec = np.zeros(nmax)
    P_vec = np.zeros(nmax)
    
    # Initialize position in f-k space
    ft = f[i]
    kt = k[j]
    
    # Pre-compute values that we'll use frequently inside loop
    hk = dk/2
    hf = df/2
    theta = np.arctan(slope)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Take line integral of fourier coefficient magnitude. 
    # The contribution of each cell to integral is equivalent to it's value
    # multiplied by the lenght of the line segment that is inside the cell. 

    L_sum = 0
    LP_sum = 0
    n = 0

    while (i < imax) and (j < jmax):

        L1 = (k[j] + hk - kt)/cos_theta
        L2 = (f[i] + hf - ft)/sin_theta

        if L1 < L2:
            L = L1
            
            L_vec[n] = L_sum + 0.5*L
            P_vec[n] = P[i,j]
            
            L_sum += L
            LP_sum += L*P[i,j]
            kt = k[j] + hk
            ft = slope*kt
            j += 1
            
        else:
            L = L2
            
            L_vec[n] = L_sum + 0.5*L
            P_vec[n] = P[i,j]
            
            L_sum += L
            LP_sum += L*P[i,j]
            ft = f[i] + hf
            kt = ft/slope
            i += 1
            
        n+=1

    integral_value = LP_sum/L_sum
    L_vec = L_vec[0:n]
    P_vec = P_vec[0:n]
    
    return(integral_value,L_sum,L_vec,P_vec)

def vector_line_integral_upstream(P,f,k,slope):
    """
    Calculate the line integral of f-k plot coefficients returned by 2D discete fourier transform 
    over the slope line associated with a specified upstream (i.e., negative slope) speed of sound.
    
    Return a vector with value of power spectrum at different lengths along line. 
    
    Algorithm is based on the one described in Appendix A of Ahmed Abukhamsin's PhD thesis
    (url: http://purl.stanford.edu/db484pp7251). 
    
    param: P: Power spectrum (f-k plot). Rows correspond to frequency, columns to wavenumber. 
    param: f: frequency values [1/s] associated with rows of A. 
    param: k: wavenumber values [1/m] associated with columns of A. 
    param: slope: speed of sound guess over which to integrate [m/s]. 
    returns: integral_value: value of line integral, normalized by length of line. 
    """
    
    # Get spacing between f-k plot grid cells 
    dk = k[1] - k[0]
    df = f[1] - f[0]
    
    # Start at origin
    i = np.argmin(np.abs(f))
    j = np.argmin(np.abs(k))
    
    # Specify stopping criteria
    imax = len(f)
    jmin = 0
    
    # Determine maximum number of grid cells that line could potentially pass through
    # Multiply by factor of 1.25 to give ourselves extra room
    nmax = int(1.25*np.sqrt((imax-i)**2 + (j-jmin)**2))
    L_vec = np.zeros(nmax)
    P_vec = np.zeros(nmax)
    
    # Initialize position in f-k space
    ft = f[i]
    kt = k[j]
    
    # Pre-compute values that we'll use frequently inside loop
    hk = dk/2
    hf = df/2
    theta = np.arctan(-1*slope)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Take line integral of fourier coefficient magnitude. 
    # The contribution of each cell to integral is equivalent to it's value
    # multiplied by the lenght of the line segment that is inside the cell.

    L_sum = 0
    LP_sum = 0
    n = 0

    while (i < imax) and (j > jmin):

        L1 = (kt - k[j] + hk)/cos_theta
        L2 = (f[i] + hf - ft)/sin_theta

        if L1 < L2:
            L = L1
            
            L_vec[n] = L_sum + 0.5*L
            P_vec[n] = P[i,j]
            
            L_sum += L
            LP_sum += L*P[i,j]
            kt = k[j] - hk
            ft = slope*kt
            j -= 1
        else:
            L = L2
            
            L_vec[n] = L_sum + 0.5*L
            P_vec[n] = P[i,j]
            
            L_sum += L
            LP_sum += L*P[i,j]
            ft = f[i] + hf
            kt = ft/slope
            i += 1
        
        n+=1

    integral_value = LP_sum/L_sum
    L_vec = L_vec[0:n]
    P_vec = P_vec[0:n]
    
    return(integral_value,L_sum,L_vec,P_vec)
    
