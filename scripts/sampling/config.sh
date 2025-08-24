### workflow parameters
### Reed Essick (reed.essick@gmail.com)

#-------------------------------------------------

### basic workflow parameters

#BASEDIR="$PWD"
BASEDIR="./many-draws"

#------------------------

NUM_GRID_PC2=500

MIN_GRID_PC2="1e10"
MAX_GRID_PC2="1e17"

#------------------------

NUM_EOS=10000       # the number of EoS drawn from the conditioned prior

NUM_EOS_SAMPLES=100 # number of samples for plot
NUM_REFERENCE_SAMPLES=25 # the number of reference EoS overlaid within (pressure, energy_density) plots

SAMPLE_ALPHA=0.1

#------------------------

EOS_COLUMN="eos"

#-------------------------------------------------

### the sets of (low-density) theories to analyze

RMFS=""

RMFS="$RMFS  1109"
RMFS="$RMFS 11233"

#-------------------------------------------------

### the crust model

CRUST="processes_from_literature/ingo-bps-with-cs2c2.csv"

REFERENCE_PRESSUREC2="1e11" # the pressure at which we stitch to the crust

#-------------------------------------------------

### the maximum pressures up to which we trust the (low-density) theory

MAXPC2S=""

#MAXPC2S="$MAXPC2S 1e11"  # should be (almost?) entirely crust below this pressure
#MAXPC2S="$MAXPC2S 1e12"
MAXPC2S="$MAXPC2S 3e12"
#MAXPC2S="$MAXPC2S 1e13"
MAXPC2S="$MAXPC2S 3e13"
#MAXPC2S="$MAXPC2S 1e14"

#------------------------

### which "agnostic" models from the literature we use

COMPDIR="processes_from_literature"

COMPS=""

COMPS="$COMPS hadagn"
COMPS="$COMPS hypagn"
COMPS="$COMPS qrkagn"

#------------------------

### used to help condition the matching between low-density theory and high-density agnostic models

#SMOOTHING_LENGTH=5.0
#SMOOTHING_LENGTH_NAME="05d000"

SMOOTHING_LENGTH=1.0
SMOOTHING_LENGTH_NAME="01d000"

#-----------

SMOOTHING_SIGMA=0.01
SMOOTHING_SIGMA_NAME="00d010"
