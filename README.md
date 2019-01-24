# gnilc
Needlet Internal Linear Combination method for CMB Component separation

# Usage
Syntax: 

python nilc.py -c gnilc.ini [--options]

Edit "gnilc.ini" to set the parameters.
Use options "--param param_parsed" to overwrite the parameters. (See gnilc.ini for explanations.)

# References

[A full sky, low foreground, high resolution CMB map from WMAP](https://arxiv.org/abs/0807.0773)

[Foreground component separation with generalised ILC](https://arxiv.org/abs/1103.1166)

# Notes 
Input HealPix maps decomposed into needlet components on different angular scales defined by window functions 
in multipole space. Needlets are localized both in pixel space and in multipole space which makes them suitable 
for cleaning the foregrounds since the foreground components are localized and have varying powers in different 
angular scales.
Internal Linear Combination (ILC) foreground cleaning method can be implemented in needlet basis for the data model
(written in vector form with each elements corresponding to a detector frequency)

xj = a sj + fj + nj

where sj is the CMB jth needlet component, a is the mixing vector that scales the power for each frequency channel
according to the emission law of the comonent (for CMB it's 1 in all elements), fj is the total foreground component,
and nj is the noise model. The ILC filtered CMB component is given by

sj = (aT R-1ja)-1 aT R-1j xj

where the covariance matrix Rj = <xjxTj>D are calculated over regions D that are determined by the local properties
of the input maps.

If a noise covariance model Nj is available, a foreground ILC filter can be constructed from the noise-whitened covariance
matrix Fj = (N-1/2j)T Rj N-1/2j and the total foreground component can be estimated by filtering the modes that correlate
between frequencies and projecting them on the subspace that is orthogonal to the CMB modes.

fj = A (AT F-1jA)-1 AT F-1j xj

where A is the estimated foreground mixing matrix.
