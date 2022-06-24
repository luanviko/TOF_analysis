import numpy as np, uproot, matplotlib.pyplot as plt, sys

def plot_waveform(event, waveform, start_rise_time, end_rise_time, a, b, timing):
    # NOTICE: mkdir ./linear_test BEFORE USING THIS FUNCTION.
    # Plot a waveform with linear interpolation of rise time.
    # PARAMETER: _event_: index for file name.
    #            _waveform_: samples to plot
    #            _start_rise_time_: start of interpolation range/ rise time.
    #            _end_rise_time_: end of interpolation range/ rise time.
    #            _a_ and _b_: linear parameters 
    #            _timing_: horizontal position of "halfway" point
    # OUTPUT: png file of name ./linear_test/waveform-1.png, for event 1.
    x = np.arange(0,512)
    fig, axi = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
    axi.set_xlim([0,512])
    axi.set_xlabel("Time (ns)")
    axi.set_ylabel("Amplitude (mV)")
    axi.plot(x,waveform[0:512])
    axi.plot(x[start_rise_time], waveform[start_rise_time], 'x')
    axi.plot(x[end_rise_time], waveform[end_rise_time], 'x')
    y = lambda x: a + b*x
    x_interpolation = np.arange(start_rise_time, end_rise_time+0.1, 0.1)
    axi.plot(timing, y(timing), 'x')
    axi.plot(x_interpolation, y(x_interpolation))
    plt.tight_layout()
    plt.savefig(f"./linear_test/waveform-{event}.png")

def walk_forward(waveform, reference):
    ## Walk forward on the _waveform_ samples.
    # Finds first sample _j_ below _reference_ values.
    # Returns sample number _j-1_.
    try:
        sample_value = waveform[0]
        j = 0
        while ((sample_value > reference) and (j < len(waveform)-1)):
            j += 1
            sample_value = waveform[j]
        return j-1
    except IndexError:
        return 10000

def linear_interpolation(x0, y0, x1, y1):
    ## Find the linear _a_ and angular _b_ parameters.
    # Provide initial (_x0_,_y0) and final (_x1_, _y1_) coordiantes.
    # Returns _a_ and _b_. 
    b = (y1-y0)/(x1-x0)
    a = y0-x0*b
    return a, b

def sum_samples(waveform, A, B, horizontalScaleFactor):
    # Simple sum over waveform sample from A to B times horizontal bin width.
    # Horizontal bin width is defined by horizontalScaleFactor.
    charge = 0.
    for i in range(A, B):
        charge += waveform[i]*horizontalScaleFactor
    return charge

def find_pulse_information(event, waveform, start_rise, end_rise, percentage, nSamples, noise_threshold, horizontalScaleFactor):
    # Find timing of PMT pulse by linear interpolation of the rise time.
    # Parameters. Find amplitude, and charge as sum of samples.
    #   _start_rise: percentage of pulse amplitude where rise time starts.
    #   _end_rise: percentage of pulse amplitude where rise time ends.
    #   _percentage_: "halfway" of rise time 
    imax00_global = np.argmax(-1.*waveform[0:nSamples])
    ymax00 = waveform[imax00_global]
    rise_amplitude = (end_rise-start_rise)*ymax00
    y_end = end_rise*ymax00
    i_end = walk_forward(waveform[0:imax00_global],y_end)
    y_start = start_rise*ymax00
    if y_start > noise_threshold:
        y_start = noise_threshold
    i_start = walk_forward(waveform[0:imax00_global],y_start)
    if (i_end == 10000 or i_start== 10000):
        print("Warning: not able to retrieve waveform ", event)
        return 10000, 10000, 0, 0
    a, b = linear_interpolation(i_start, y_start, i_end, y_end)
    imax00_CFD = (percentage*rise_amplitude-a)/b
    charge00 = sum_samples(waveform[0:nSamples], 0, nSamples, horizontalScaleFactor)
    # plot_waveform(event, waveform, i_start, i_end, percentage, a, b, imax00_CFD)
    return imax00_CFD, imax00_global, ymax00, charge00

def findBaseline_uproot(table, nWaveforms, sampleCap):
    ## Sum over the *sampleCap* first samples
    #  of the waveform to find baseline.
    #  Average this value over nWaveforms percentage
    #  of the total waveforms.
    baseline = 0.
    sumBaselineErrorSquared = 0.
    sampleWaveforms = int(nWaveforms*len(table) )
    for i in range(0, sampleWaveforms ):
        waveform = table[i]
        sumSamples = 0.
        sumSamplesErrorSquared = 0.
        for j in range(0,sampleCap):
            sumSamples += waveform[j]
            sumSamplesErrorSquared += (0.05*waveform[j])**2
        waveformBaselineError = np.sqrt(sumSamplesErrorSquared)/sampleCap
        baseline += sumSamples/sampleCap
        sumBaselineErrorSquared += waveformBaselineError**2
    return baseline/sampleWaveforms, np.sqrt(sumBaselineErrorSquared)/sampleWaveforms

def find_channel_baseline(waveforms00, verticalScaleFactor):
    # TO_DO: Fix hard-coded percentage of waveforms (0.01)
    #        Fix hard-coded numper of samples (20)
    baseline00, baseline_error00 = findBaseline_uproot(waveforms00, 0.01, 20)
    baseline00 = baseline00*verticalScaleFactor
    baselineError00 = baseline_error00*verticalScaleFactor
    return baseline00, baselineError00

def save_txt(run_number, key, CFD_times, global_times, amplitudes, charges):
    # Save timing, amplitude and charge to a txt/csv file.
    # File name: ./pulse_information-99-dt5743_wave05.txt for run 00 and channel 5.
    with open(f"./pulse_information-{run_number}-{key}.txt","w") as file_output:
        for i in range(0, len(CFD_times)):
            file_output.write(f"{CFD_times[i]}, {global_times[i]}, {amplitudes[i]}, {charges[i]}\n")


## START OF MAIN SCRIPT

# Digitizer specs
mppc_DIGITIZER_FULL_SCALE_RANGE = 2.5 # Vpp
mppc_DIGITIZER_RESOLUTION       = 12 # bits
mppc_DIGITIZER_SAMPLE_RATE      = 3200000000 # S/s
digiCounts = 2.**mppc_DIGITIZER_RESOLUTION
nSamples = int(512)

# Scale factors
verticalScaleFactor = 1.e3*mppc_DIGITIZER_FULL_SCALE_RANGE/digiCounts; # V/bank
horizontalScaleFactor = 1.e9/mppc_DIGITIZER_SAMPLE_RATE #ns/sample

percentage=0.5
start_rise = 0.1
end_rise = 0.9
noise_threshold = -10. #mV

if len(sys.argv) != 3:
    print("How to use: python TOF_analysis_8channels.py _output.root_ _run_number_")
    sys.exit(1)

# Retrieve run number 
run_number = sys.argv[2]

# Open root file and get waveforms
file_name = sys.argv[1]
with uproot.open(file_name) as fin:
    table = fin['waveform_tree']
    waveforms= table.arrays(['dt5743_wave00', 'dt5743_wave01', 'dt5743_wave02', 'dt5743_wave03', 'dt5743_wave04', 'dt5743_wave05', 'dt5743_wave06', 'dt5743_wave07'], library="np")

# Loop over each channel
for key in waveforms.keys():
    
    # Initialize arrays to store information
    CFD_times = np.array([],"float")
    global_times = np.array([],"float")
    amplitudes = np.array([],"float") 
    charges = np.array([],"float")

    # Find baseline, timing, amplitude and charge.
    # Save each waveform's information to arrays above.
    channel_waveforms = waveforms[key]
    print("Analyzing branch '"+key+"'.")
    baseline, baseline_error = find_channel_baseline(channel_waveforms, verticalScaleFactor)
    channel_waveforms = channel_waveforms*verticalScaleFactor-baseline
    print("Baseline: ", baseline)
    for i in range(0, len(channel_waveforms)):
        print(f"Finding pulse information: {(i+1)/len(channel_waveforms):4.2f}", end="\r")
        timing_CFD, timing_global, amplitude, charge = find_pulse_information(i, channel_waveforms[i], start_rise, end_rise, percentage, nSamples, noise_threshold, horizontalScaleFactor)
        CFD_times = np.append(CFD_times, [timing_CFD])
        global_times = np.append(global_times, [timing_global])
        amplitudes = np.append(amplitudes, [amplitude])
        charges = np.append(charges, [charge])
    save_txt(run_number, key, CFD_times, global_times, amplitudes, charges)
    print("Finding pulse information: 100.00%")