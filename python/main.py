"""
Arno Annys
"""

from  propagator import propagator

examp1 = propagator(nt = 300)
examp1.shape_potential('free')
examp1.propagate()

examp2 = propagator(nt = 300)
examp2.shape_potential('circle')
examp2.propagate()

examp3 = propagator(nt = 300)
examp3.shape_potential('barrier')
examp3.propagate()


examp4 = propagator(nt = 150)
examp4.shape_potential('single_slit')
examp4.propagate()

examp5 = propagator(nt = 150)
examp5.shape_potential('double_slit')
examp5.propagate()
