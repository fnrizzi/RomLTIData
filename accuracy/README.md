
This test is for testing ROM accuracy:

- we consider a 1d parameter, the forcing period T in range (35, 65) (seconds)
- the ROM pod are computed using only data from T=35, and T=65 
- we then randomly sample the range and check how accurate the ROM is 
- we also take two samples outside of the range to check how well ROM 
does with extrapolation 

- sampling for the FOM is done using 2 seconds 
- ricker wavelet for simulatin the source signal
- each directory contains the input file used to run

- the log of the python script generating all data is [here](py.out)
- errors computed at the final step: [velocity](errors_table_vp.txt), and [stress](errors_table_vp.txt)
- we compute l2 and linf errors, both absolute and relative
- the script to parse all directoreis and create the tables with the errors is [here](parse_errors.py)

- note that some directories might not have the full data. 
We dont store here the POD mdoes because thye are too big and 
we only save here the final state for some test points, since storing the full states 
everwhere becomes too expensive
