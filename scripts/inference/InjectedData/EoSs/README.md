EoSs Used for injections

We use an RMF EoS (#338 of the 1109 RMF EoSs) given in
rmf-1109-000338.csv
I've added a crust for TOV integration

And we stitch onto a constant speed of sound extension (for
all cases here we use c_s^2 = .8) above a certain density
For example

./abht_qmcrmf2_2p0_1p1.csv stiches onto a css EoS at rho 2.0 times saturation density,
with a transition enthalpy delta e / e = 1.1

./abht_qmcrmf2_2p0_1p2.csv stitches onto a css EoS at rho 2.0 times saturation with
a transition enthalpy delta e / e = 1.2

etc...

See ./design_pt_eos.py for how we build these css extensions,

./get_macro.sh calls the universality TOV solver

